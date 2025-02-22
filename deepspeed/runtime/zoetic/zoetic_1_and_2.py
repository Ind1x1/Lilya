# Copyright 2024
# Zoetic Project (a) 2024 Leyi Ye
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import copy
import time
from deepspeed import comm as dist
from packaging import version as pkg_version
from collections import OrderedDict
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.runtime.base_optimizer import ZeROOptimizer
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler
from deepspeed.runtime.utils import (empty_cache, see_memory_usage, inf, is_model_parallel_parameter,
                                     align_dense_tensors, all_gather_dp_groups)
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils import logger
from deepspeed.utils.bwc import bwc_tensor_model_parallel_rank
from deepspeed.moe.utils import is_moe_param
from deepspeed.git_version_info import version

from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.accelerator import get_accelerator

from deepspeed.checkpoint.constants import (DS_VERSION, GROUP_PADDINGS, PARTITION_COUNT, LOSS_SCALER,
                                            SINGLE_PARTITION_OF_FP32_GROUPS, BASE_OPTIMIZER_STATE,
                                            BASE_OPTIMIZER_STATE_STEP, CLIP_GRAD, ZERO_STAGE, PARAM_SLICE_MAPPINGS)
from deepspeed.utils import link_hp_params, lazy_init_hp_params_optimizer_state
from deepspeed.checkpoint import enable_universal_checkpoint

from deepspeed.utils import groups
# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

OPTIMIZER_ALLGATHER_TIMER = 'optimizer_allgather'
OPTIMIZER_GRADIENTS_TIMER = 'optimizer_gradients'
OPTIMIZER_STEP_TIMER = 'optimizer_step'
OPTIMIZER_TIMERS = [OPTIMIZER_ALLGATHER_TIMER, OPTIMIZER_GRADIENTS_TIMER, OPTIMIZER_STEP_TIMER]
INITIAL_MICRO_STEP_ID = -1


import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def input(msg):
    return


def split_half_float_double(tensors):
    device_type = get_accelerator().device_name()
    dtypes = [
        "torch.{}.HalfTensor".format(device_type), "torch.{}.FloatTensor".format(device_type),
        "torch.{}.DoubleTensor".format(device_type), "torch.{}.BFloat16Tensor".format(device_type)
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def isclose(a, b, rtol=1e-09, atol=0.0):
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def lcm(x, y):
    from fractions import gcd  # or can import gcd from `math` in Python 3
    return x * y // gcd(x, y)


def get_alignment_padding(tensor_list, alignment):
    num_elements = sum([tensor.numel() for tensor in tensor_list])
    remainder = num_elements % alignment
    return (alignment - remainder) if remainder else remainder


def print_rank_msg(msg):
    print(f"rank {dist.get_rank()} - {msg}")


def _get_padded_tensor(src_tensor, size):
    if src_tensor.numel() >= size:
        return src_tensor
    padded_tensor = torch.zeros(size, dtype=src_tensor.dtype, device=src_tensor.device)
    slice_tensor = torch.narrow(padded_tensor, 0, 0, src_tensor.numel())
    slice_tensor.data.copy_(src_tensor.data)
    return padded_tensor


def _pad_tensor_by_size(src_tensor, pad_size, dtype, device):
    padded_tensor = torch.zeros(src_tensor.numel() + pad_size, dtype=dtype, device=device)
    padded_tensor.data[:src_tensor.numel()].copy_(src_tensor.data)
    return padded_tensor

class ZoeticZeroOptimizer(DeepSpeedZeroOptimizer):

    def __init__(self,
                 init_optimizer,
                 param_names,
                 timers,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 use_multi_rank_bucket_allreduce=True,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 mpu=None,
                 clip_grad=0.0,
                 gradient_accumulation_dtype=torch.float32,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 elastic_checkpoint=False,
                 vertin_cpu_optimizer = None):
        
        if offload_optimizer_config is not None and offload_optimizer_config.device != OffloadDeviceEnum.none:
            self.cpu_offload = True
            self.cpu_offload_pin_memory = offload_optimizer_config.pin_memory
        else:
            self.cpu_offload = False
            self.cpu_offload_pin_memory = False

        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Allgather bucket size {allgather_bucket_size}")
            logger.info(f"CPU Offload: {self.cpu_offload}")
            logger.info(f'Round robin gradient partitioning: {round_robin_gradients}')
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        self.elastic_checkpoint = elastic_checkpoint
        self.param_names = param_names
        self.mpu = mpu
        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        if not get_accelerator().is_available():
            raise SystemError("Accelerator is not detected, cannot perform low precision training (e.g., fp16, bf16).")
        self.optimizer = init_optimizer

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        # ZeRO stage 1 (False) or 2 (True)
        self.partition_gradients = partition_grads
        self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"

        self.timers = timers

        self.reduce_scatter = reduce_scatter

        self.overlap_comm = overlap_comm

        self.deepspeed_adam_offload = self.cpu_offload

        self.device = get_accelerator().current_device_name() if not self.cpu_offload else 'cpu'

        self.dp_process_group = dp_process_group
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        #expert parallel group
        self.ep_process_group = expert_parallel_group

        #data parallel group for experts
        self.expert_dp_process_group = expert_data_parallel_group

        #data parallel size for non-experts
        dp_size = dist.get_world_size(group=self.dp_process_group)

        #For MoE models this maybe different for different param group
        #It will be modified during MoE setup later in the init
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]

        self.is_gradient_accumulation_boundary = True

        # CPU-Offload requires contiguous gradients
        self.contiguous_gradients = contiguous_gradients or self.cpu_offload

        self.has_moe_layers = has_moe_layers
        if self.has_moe_layers:
            self._configure_moe_settings()
        self._global_grad_norm = 0.

        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_world_size = 1
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_world_size = mpu.get_model_parallel_world_size()
            self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)

        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = INITIAL_MICRO_STEP_ID
        self.ignore_unused_parameters = ignore_unused_parameters
        self.round_robin_gradients = round_robin_gradients

        self.extra_large_param_to_reduce = None
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients

        if self.fp16_master_weights_and_gradients:
            assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], \
            f"fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32."\
            f"Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}." \
            f"Either disable fp16_master_weights_and_gradients or enable {self.zero_stage_string} Offload with DeepSpeedCPUAdam."

        if self.reduce_scatter and self.partition_gradients:
            valid_reduce_scatter_dtypes = (torch.float16, torch.bfloat16, torch.float32)
            assert self.communication_data_type in valid_reduce_scatter_dtypes, f"{self.zero_stage_string} supports {valid_reduce_scatter_dtypes} communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, f"gradient_predivide_factor != 1.0 is not yet supported with {self.zero_stage_string} with reduce scatter enabled"
            assert self.postscale_gradients, f"pre-scale gradients is not yet supported with {self.zero_stage_string} with reduce scatter enabled"
        #######################################################################################
        self.zoetic_partition_of_fp32_groups = []
        self.zoetic_partition_of_fp32_groups_local = []
        #######################################################################################
        # param flattened by groups
        self.bit16_groups = []
        self.bit16_groups_flat = []

        # param partitioned by data parallel degree
        # this will contain a list of equal sized tensors
        # each of which will be updated by a different process
        self.parallel_partitioned_bit16_groups = []

        # a single 32-bit partition of the parallel partitioned parameters
        # that this process will update
        self.single_partition_of_fp32_groups = []

        # param partition info

        # These are the parameters in each group that will not be updated by this process directly
        self.params_not_in_partition = []

        # These are the parameters that will be updated by this process directly
        self.params_in_partition = []

        # Offset from the first parameter in the self.params_in_partition
        # the parameter boundaries may not align with partition boundaries
        # so we need to keep track of the offset
        self.first_offset = []

        # number of elements per partition in each group
        self.partition_size = []

        # align nccl all-gather send buffers to 4-byte boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        assert (
            allgather_bucket_size % self.nccl_start_alignment_factor == 0
        ), f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} "

        self.all_reduce_print = False
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self.gradient_accumulation_dtype = gradient_accumulation_dtype

        if self.dtype != self.gradient_accumulation_dtype:
            self.use_separate_grad_accum = True
        else:
            self.use_separate_grad_accum = False
        if self.use_separate_grad_accum and not self.partition_gradients:
            self.use_grad_accum_attribute = True
        else:
            self.use_grad_accum_attribute = False

        self.round_robin_bit16_groups = []
        self.round_robin_bit16_indices = []
        self.round_robin_bit16_meta = []

        # Use different parallel to do all_to_all_reduce related things
        # padding on each partition for alignment purposes
        self.groups_padding = []
        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            #########################################################################
            checkpoint_id = self.replica_rank(partition_id)
            #########################################################################

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            trainable_parameters = []
            for param in param_group['params']:
                if param.requires_grad:
                    param.grad_accum = None
                    trainable_parameters.append(param)
            # bit16_groups contains all the parameters in the model
            self.bit16_groups.append(trainable_parameters)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")
            # move all the parameters to cpu to free up GPU space for creating flat buffer

            # Create temp CPU param copies, free accelerator tensors
            orig_group_numel = 0
            for param in self.bit16_groups[i]:
                orig_group_numel += param.numel()
                param.cpu_data = param.data.cpu()
                param.data = torch.empty(1).to(param.device)

            empty_cache()
            see_memory_usage(f"After moving param group {i} to CPU", force=False)

            # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
            # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
            # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
            # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
            if self.round_robin_gradients:
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                    self.bit16_groups[i], dist.get_world_size(group=self.real_dp_process_group[i]))
                # robin tensors 返回的round_robin_tensors是按照round_robin_indices的顺序对bit16_groups[i]重新排列的
            else:
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))

            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)

            # Create meta tensors list, ordered according to round_robin_tensors
            meta_tensors = []
            for param in round_robin_tensors:
                meta_tensors.append(torch.zeros_like(param.cpu_data, device="meta"))
            self.round_robin_bit16_meta.append(meta_tensors)

            # create flat buffer in CPU
            flattened_buffer = self.flatten_dense_tensors_aligned(
                self.round_robin_bit16_groups[i],
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]),
                use_cpu_data=True)

            # free temp CPU params
            for param in self.bit16_groups[i]:
                del param.cpu_data

            # Move CPU flat tensor to the accelerator memory.
            self.bit16_groups_flat.append(flattened_buffer.to(get_accelerator().current_device_name()))
            del flattened_buffer

            see_memory_usage(f"After flattening and moving param group {i} to GPU", force=False)

            # Record padding required for alignment last partition have padding record the padding size
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                padding = self.bit16_groups_flat[i].numel() - orig_group_numel
            else:
                padding = 0
            self.groups_padding.append(padding)

            if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
                see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=False)

            # set model bit16 weight to slices of flattened buffer
            self._update_model_bit16_weights(i)

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            # 切分flat tensor
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i) 
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)
            self.zoetic_partition_of_fp32_groups_local.append(data_parallel_partitions)

            # verify that data partition start locations are 4-byte aligned
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

            # A partition of the fp32 master weights that will be updated by this process.
            # Note that the params in single_partition_of_fp32_groups is cloned and detached
            # from the origin params of the model.
            if not fp16_master_weights_and_gradients:
                weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().float().detach()
            else:
                weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().half().detach()
            
            #########################################################################
            # 最终版本这里应该是个for循环 以创建复数副本
            # partition_id is the shard rank of local 
            if not fp16_master_weights_and_gradients:
                checkpoint_partition = self.parallel_partitioned_bit16_groups[i][checkpoint_id].clone().float().detach().cpu().share_memory_()
            else:
                checkpoint_partition = self.parallel_partitioned_bit16_groups[i][checkpoint_id].clone().float().detach().cpu().share_memory_()

            self.zoetic_partition_of_fp32_groups.append(checkpoint_partition)
            self.zoetic_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it
            
            ##########################################################################
            if self.cpu_offload:
                weights_partition = get_accelerator().pin_memory(weights_partition)

            self.single_partition_of_fp32_groups.append(weights_partition)

            # Set local optimizer to have flat params of its own partition.
            # After this, the local optimizer will only contain its own partition of params.
            # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
            self.single_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
                self.round_robin_bit16_groups[i], partition_size, partition_id)

            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

        #########################################################################################################
        # local replica
        self.local_optimizer_param_groups = copy.deepcopy(self.optimizer.param_groups)
        for i, param_group in enumerate(self.local_optimizer_param_groups):
            param_group['params'] = [param.detach().clone().cpu().share_memory_() for param in param_group['params']]
            for tensor in param_group['params']:
                tensor.requires_grad = True
            if 'bias_correction' not in param_group:
                param_group['bias_correction'] = True
        for i, param_group in enumerate(self.local_optimizer_param_groups):
            for param in param_group['params']:
                single_grad_partition = torch.zeros(int(param.numel()),
                                                    dtype = self.single_partition_of_fp32_groups[i].dtype,
                                                    device = 'cpu')
                single_grad_partition.share_memory_()
                param.grad = single_grad_partition
        #########################################################################################################
        self.remote_optimizer_param_groups = copy.deepcopy(self.optimizer.param_groups)
        for i, param_group in enumerate(self.remote_optimizer_param_groups):
             param_group['params'] = [self.zoetic_partition_of_fp32_groups[i]]
        for i, param_group in enumerate(self.remote_optimizer_param_groups):
            for param in param_group['params']:
                single_grad_partition = torch.zeros(int(param.numel()),
                                                    dtype = self.zoetic_partition_of_fp32_groups[i].dtype,
                                                    device = 'cpu')
                single_grad_partition.share_memory_()
                param.grad = single_grad_partition
        #########################################################################################################
        # 至此我们获得了两个 两个optimizer self.remote_optimizer_param_groups slef.local_optimizer_param_groups
        #########################################################################################################
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.use_multi_rank_bucket_allreduce = use_multi_rank_bucket_allreduce
        self.allgather_bucket_size = int(allgather_bucket_size)

        self.reduction_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream()
        #self.copy_grad_stream = get_accelerator().Stream()
        self.callback_queued = False

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None
        self.ipg_bucket_has_moe_params = False

        # simplified param id
        self.param_id = {}

        #interesting code: unique ids being assigned to individual parameters
        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True

        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        if self.cpu_offload:
            self.accumulated_grads_in_cpu = {}
            self.norm_for_param_grads = {}
            self.local_overflow = False
            self.grad_position = {}
            self.temp_grad_buffer_for_cpu_offload = torch.zeros(largest_param_numel,
                                                                device=self.device,
                                                                dtype=self.dtype)
            if self.cpu_offload_pin_memory:
                self.temp_grad_buffer_for_cpu_offload = get_accelerator().pin_memory(
                    self.temp_grad_buffer_for_cpu_offload)
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                                device=get_accelerator().current_device_name(),
                                                                dtype=self.dtype)
            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])

        # mapping from parameter to partition that it belongs to
        self.param_to_partition_ids = {}

        # stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        # number of grads in partition that still need to be computed
        self.remaining_grads_in_partition = {}

        # total number of grads in partition
        self.total_grads_in_partition = {}

        # stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        # stores the offset at which a parameter gradient needs to be inserted in a partition
        self.grad_partition_insertion_offset = {}

        # the offset in the gradient at which it must be inserted at the beginning of the partition
        self.grad_start_offset = {}

        # will store the averaged gradients required by this partition
        self.averaged_gradients = {}

        # For cpu_offload, will store the averaged gradients required by this partition
        self.offload_gradient_dict = {}

        # store index of first parameter in each partition
        self.first_param_index_in_partition = {}

        # initializes all data structures for implementing gradient partitioning
        self.initialize_gradient_partitioning_data_structures()

        # resets the data structure value for the next backward propagation
        self.reset_partition_gradient_structures()

        # creates backward hooks for gradient partitioning
        self._grad_acc_hooks = []
        if self.partition_gradients or self.overlap_comm:
            self.create_reduce_and_remove_grad_hooks()

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        # we may have a way of fusing dynamic scale. Do not support for now
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic

        if self.dtype != torch.float16:
            # Only fp16 should use dynamic loss scaling
            assert self.loss_scaler.cur_scale == 1.0
            assert not self.dynamic_loss_scale

        see_memory_usage("Before initializing optimizer states", force=True)
        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)

        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")

        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)

        self._link_all_hp_params()
        self._hp_optimizer_states_linked = False

        self._enable_universal_checkpoint()
        self._param_slice_mappings = self._create_param_mapping()
        ########################################################################################################

        self.dist_world_size = dist.get_world_size()
        self.dist_local_rank = dist.get_local_rank()
        self.dist_rank = dist.get_rank()

        # self.vertin_param_groups = copy.deepcopy(self.optimizer.param_groups)

        #zoetic
        self.zoetic_FLAG = False
        # dafault replica
        self.zoetic_replica = 1

        self.zoetic_checkpoint_group = self.replica_checkpoint_group

        self.zoetic_buffer = None
        self.zoetic_index = 0
        self.zoetic_offset = 0
        self.zoetic_numel = 0

        #vertin
        self.vertin_pool = ProcessPoolExecutor(1)
        self.vertin_lock = mp.Lock()

        self.gpu_vertin_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream()
        self.cpu_vertin_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream()

        # zoetic 
        self.zoetic_bit16_group = []
        self.zoetic_interwine = []
        self.vertin_grad = []
        self.zoetic_bucket_size = 50000

    # import re-init zoetic param group
    # def zoetic_param_group_init(self):
    #     self.vertin_param_groups = copy.deepcopy(self.optimizer.param_groups)
    #     vertin_trainable_parameters =[]
    #     for i, param_group in enumerate(self.vertin_param_groups):
    #         for param in param_group['params']:
    #             if param.requires_grad == None:
    #                 param.grad_accm = None
    #                 vertin_trainable_parameters.append(param)
    #         self.zoetic_bit16_group.append(vertin_trainable_parameters)

    def replica_checkpoint_group(self):
        # [0,0,0,0,0,0,0,0] 
        tensor = torch.zeros(self.dist_world_size)
        # 设置 dist_rank 位置的元素为 dist_local_rank
        tensor[self.dist_rank] = self.dist_local_rank
        dist.all_reduce(tensor, group=self.dp_process_group)
        # [0,1,2,3,0,1,2,3]
        machine_count = torch.sum(tensor == 0).item()
        assert machine_count > 1, "Warning: Zoetic avaliable for multi machine request machine_count should be greater than 2"
        # checkpoint_group = self.initialize_checkpoint_placement(rank=self.dist_rank,machine_count=machine_count,replice=self.zoetic_replica)
        # self.zoetic_checkpoint_group = self.initialize_checkpoint_group(checkpoint_group)
        rank = dist.get_rank()
        if rank == 0 or rank == 1 :  # rank 0 和 1 成为一组
            return dist.new_group([0, 1])
        else:  # rank 2 和 3 成为一组
            return dist.new_group([2, 3])
    

    def initialize_checkpoint_placement(self, rank, machine_count, replice):
        pass
    
    def initialize_checkpoint_group(self,checkpoint_group):
        pass
    
    def replica_rank(self, partition_id):
        if partition_id == 0:
            return 1
        elif partition_id == 1:
            return 0
        elif partition_id == 2:
            return 3
        elif partition_id == 3:
            return 2
        
    # 重构backward 主要是为了加一个 zoetic buf 以实现梯度啊传递
    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1

        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=get_accelerator().current_device_name())
            self.ipg_buffer.append(buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=get_accelerator().current_device_name())
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0

        self._vertin_create_buf(self.zoetic_flag, self.zoetic_bucket_size)

        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

        # Only for Stage 1, Mode 2
        if self.use_grad_accum_attribute:
            self.fill_grad_accum_attribute()


    def _vertin_create_buf(self, zoetic_flag = False, bucket_size = 50000):
        if zoetic_flag:
            self.zoetic_buffer = []
            buf_0 = torch.empty(bucket_size, dtype=self.dtype,device=get_accelerator().current_device_name())
            buf_1 = torch.empty(bucket_size, dtype=self.dtype,device=get_accelerator().current_device_name())
            self.zoetic_buffer.append([buf_0,buf_1])
            buf_2 = torch.empty(bucket_size, dtype=self.dtype,device=get_accelerator().current_device_name())
            buf_3 = torch.empty(bucket_size, dtype=self.dtype,device=get_accelerator().current_device_name())
            self.zoetic_buffer.append([buf_2,buf_3])
            self.zoetic_index = 0
    
    # hook 
    # 考虑到在acc的时候我们应该只需要取最终的梯度，所以我觉得应该跨过reduce_ipg_grads

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = INITIAL_MICRO_STEP_ID

        see_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        if self.dtype == torch.float16:
            self.check_overflow()

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients')

            for timer in OPTIMIZER_TIMERS:
                self.timers(timer).start()
                self.timers(timer).stop()
            return

        # Step 1:- Calculate gradient norm using bit-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale
        see_memory_usage('After norm before optimizer')

        # Step 2:- run optimizer and upscaling simultaneously
        for i, group in enumerate(self.bit16_groups):
            self.timers(OPTIMIZER_GRADIENTS_TIMER).start()
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i)

                # Disabled, this is not currently working
                #from deepspeed.ops.adam import DeepSpeedCPUAdam
                #if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                #    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                #    fp32_partition = self.single_partition_of_fp32_groups[i]
                #    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(
                    fp32_partition.to(get_accelerator().current_device_name()).data)

                self.timers(OPTIMIZER_STEP_TIMER).stop()
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
                assert single_grad_partition.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None

                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()

                # Step 3:- run the optimizer if no offloading
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i)
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                ############################获得local checkpoint grad########################
                # del single_grad_partition
                self.vertin_grad.append(single_grad_partition)
                self.zoetic_async_copy_grad_from_gpu(i, single_grad_partition) 
                ############################
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.timers(OPTIMIZER_STEP_TIMER).stop()

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)
        self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.timers.log(OPTIMIZER_TIMERS)
        see_memory_usage('After zero_optimizer step')

        return
    

    def zoetic_async_copy_grad_from_gpu(self, id ,grad):
        with get_accelerator().stream(self.cpu_vertin_stream):
            dest_tensor = self.local_optimizer_param_groups[id]['params'][0].grad
            dest_tensor.copy_(grad, non_blocking=True)

    def zoetic_comm_interwin(self):
        """
        accoding to 
        """
        self.zoetic_interwine.append(self.vertin_grad) # list [tensor1,tensor2,tensor3,tensor4]
        self.vertin_grad = None

    # def zoetic_all_gather_grad(self):
    #     #  这里 0 是有误的
    #     self.zoetic_comm_interwine() # 切分全尺寸张量
    #     zoetic_rank = dist.get_rank(self.zoetic_checkpoint_group)
    #     for tensor in enumerate(self.zoetic_interwin):
    #         offset = 0
    #         block_size = self.zoetic_bucket_size
    #         while offset+block_size < tensor.numel():
    #             dist.all_gather(self.zoetic_buffer[self.zoetic_index],tensor[offset:offset+block_size],group=self.zoetic_checkpoint_group)
    #             self.remote_optimizer_param_groups[0]['params'][0].grad[offset:offset+block_size].copy_(self.zoetic_buffer[self.zoetic_index][1-zoetic_rank])
    #             offset += block_size
    #         if offset < tensor.numel():
    #             laster_tensors = [torch.empty_like(tensor[offset:]) for _ in range(dist.get_world_size(self.zoetic_checkpoint_group))]
    #             dist.all_gather(laster_tensors,tensor[offset:],group=self.zoetic_checkpoint_group)
    #             self.remote_optimizer_param_groups[0]['params'][0].grad[offset:].copy_(laster_tensors[1-zoetic_rank])

    #     self.zoetic_interwine = []

    def zoetic_all_gather_grad(self):
        zoetic_rank = dist.get_rank(self.zoetic_checkpoint_group)
        for tensor in self.vertin_grad:
            offset = 0
            block_size = self.zoetic_bucket_size
            while offset+block_size < tensor.numel():
                dist.all_gather(self.zoetic_buffer[self.zoetic_index],tensor[offset:offset+block_size],group=self.zoetic_checkpoint_group)
                self.remote_optimizer_param_groups[0]['params'][0].grad[offset:offset+block_size].copy_(self.zoetic_buffer[self.zoetic_index][1-zoetic_rank])
                offset += block_size
            if offset < tensor.numel():
                laster_tensors = [torch.empty_like(tensor[offset:]) for _ in range(dist.get_world_size(self.zoetic_checkpoint_group))]
                dist.all_gather(laster_tensors,tensor[offset:],group=self.zoetic_checkpoint_group)
                self.remote_optimizer_param_groups[0]['params'][0].grad[offset:].copy_(laster_tensors[1-zoetic_rank])
        self.vertin_grad = []
            
    # def zoetic_all_gather_grad(self):
    #     zoetic_rank = dist.get_rank(self.vertin_checkpoint_group)
    #     print(self.vertin_grad)
    #     if self.vertin_grad ==[]:
    #         return
    #     for tensor in self.vertin_grad:
    #         offset = 0
    #         block_size = self.zoetic_bucket_size
    #         while offset+block_size < tensor.numel():
    #             index = self.zoetic_index
    #             self.cpu_vertin_stream.synchronize()
    #             with get_accelerator().stream(self.gpu_vertin_stream):
    #                 dist.all_gather(self.zoetic_buffer[index],tensor[offset:offset+block_size],group=self.vertin_checkpoint_group)
    #                 #self.remote_optimizer_param_groups[0]['params'][0].grad[offset:offset+block_size].copy_(self.zoetic_buffer[self.zoetic_index][1-zoetic_rank])

    #             self.gpu_vertin_stream.synchronize()
                
    #             with get_accelerator().stream(self.cpu_vertin_stream):
    #                 self.remote_optimizer_param_groups[0]['params'][0].grad[offset:offset+block_size].copy_(self.zoetic_buffer[index][1-zoetic_rank])
    #             offset += block_size
    #             self.zoetic_index = 1 - self.zoetic_index

    #         self.cpu_vertin_stream.synchronize()
    #         self.gpu_vertin_stream.synchronize()
    #         with get_accelerator().stream(self.gpu_vertin_stream):
    #             if offset < tensor.numel():
    #                 laster_tensors = [torch.empty_like(tensor[offset:]) for _ in range(dist.get_world_size(self.vertin_checkpoint_group))]
    #                 dist.all_gather(laster_tensors,tensor[offset:],group=self.vertin_checkpoint_group)
    #                 self.remote_optimizer_param_groups[0]['params'][0].grad[offset:].copy_(laster_tensors[1-zoetic_rank])
    #             print_rank_msg(self.remote_optimizer_param_groups[0]['params'][0].grad)
    #             print_rank_msg(self.remote_optimizer_param_groups[0]['params'][0].grad.size())
    #             self.vertin_grad = []
    
    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = INITIAL_MICRO_STEP_ID

        see_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        if self.dtype == torch.float16:
            self.check_overflow()

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients')

            for timer in OPTIMIZER_TIMERS:
                self.timers(timer).start()
                self.timers(timer).stop()
            return

        # Step 1:- Calculate gradient norm using bit-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale
        see_memory_usage('After norm before optimizer')

        # Step 2:- run optimizer and upscaling simultaneously
        for i, group in enumerate(self.bit16_groups):
            self.timers(OPTIMIZER_GRADIENTS_TIMER).start()
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i)

                # Disabled, this is not currently working
                #from deepspeed.ops.adam import DeepSpeedCPUAdam
                #if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                #    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                #    fp32_partition = self.single_partition_of_fp32_groups[i]
                #    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(
                    fp32_partition.to(get_accelerator().current_device_name()).data)

                self.timers(OPTIMIZER_STEP_TIMER).stop()
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
                assert single_grad_partition.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None

                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()

                # Step 3:- run the optimizer if no offloading
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i)
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                del single_grad_partition
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.timers(OPTIMIZER_STEP_TIMER).stop()

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)
        self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.timers.log(OPTIMIZER_TIMERS)
        see_memory_usage('After zero_optimizer step')
        return

