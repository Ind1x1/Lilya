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


####

# model_engine, optimizer, _, _ = deepspeed.initialize(
#         args=args,
#         model=model,
#         optimizer=optimizer,
#         config=ds_config
#     )

####

import torch.multiprocessing as mp

from deepspeed.runtime.engine import DeepSpeedEngine

class ZoeticProcess(mp.Process):
    def __init__(self,
                 param_local,
                 param_remote,
                 grad_list_local,
                 grad_list_remote,
                 stop_event,
                 update_flag,
                 update_group_no,
                 remote_lock,
                 local_lock):
        super().__init__()

        self.local_optimizer_param_groups = param_local
        self.remote_optimizer_param_groups = param_remote

        self.local_optimizer_param_groups_grad = grad_list_local
        self.remote_optimizer_param_groups_grad = grad_list_remote
        self.remote_lock = remote_lock
        self.local_lock = local_lock

        self.link_param_grad(self.local_optimizer_param_groups, self.local_optimizer_param_groups_grad)
        self.link_param_grad(self.remote_optimizer_param_groups, self.remote_optimizer_param_groups_grad)

        self.stop_event = stop_event
        self.update_flag = update_flag
        self.update_group_no = update_group_no
        from deepspeed.ops.vertin import SonnetVertinCPUAdam
        self.vertin_optimizer = SonnetVertinCPUAdam(self.local_optimizer_param_groups)    
    
    def link_param_grad(self, param_groups, grad_groups):
        for i, param_group in enumerate(param_groups):
            for j, param in enumerate(param_group['params']):
                param.grad = grad_groups[i][j] 

    def run(self):
        while not self.stop_event.is_set():
            with self.update_flag:
                self.update_flag.wait()
                id = self.update_group_no.value
                with self.local_lock:
                    original_param_groups = self.vertin_optimizer.param_groups
                    self.vertin_optimizer.param_groups = [self.local_optimizer_param_groups[id]]
                    print(self.vertin_optimizer.param_groups[0]['params'][0].grad)
                    self.vertin_optimizer.step()
                    self.vertin_optimizer.param_groups = original_param_groups
                    print(self.vertin_optimizer.state_dict())
                    
    # def _step(self, id):
    #     for i in range(id):
    #         print(f"run {id}")
    #         print("run step once ")
    #         original_param_groups = self.vertin_optimizer.param_groups
    #         # local update
    #         self.vertin_optimizer.param_groups = [self.local_optimizer_param_groups[i]]
    #         self.vertin_optimizer.step()
    #         # remote update 
    #         self.vertin_optimizer.param_groups = [self.remote_optimizer_param_groups[i]]
    #         self.vertin_optimizer.step()

    #         self.vertin_optimizer.param_groups = original_param_groups


class ZoeticEngine:
    def __init__(self, 
                 engine):
        self.engine = engine
        self.optimizer = engine.optimizer

        self.zoetic_flag = self.optimizer.zoetic_FLAG

        if self.zoetic_flag:
            mp.set_start_method('spawn')
            self.zoetic_update_flag = self.optimizer.zoetic_update_flag     # imple flag
            self.zoetic_group_no = self.optimizer.zoetic_group_no           # update id
            self.zoetic_stop_event = self.optimizer.zoetic_stop_event       # stop flag
            
            self.zoetic_local_lock = self.optimizer.zoetic_local_lock       # lock
            self.zoetic_remote_lock = self.optimizer.zoetic_remote_lock     # lock

            self.local_optimizer_param_groups = self.optimizer.local_optimizer_param_groups
            self.remote_optimizer_param_groups = self.optimizer.remote_optimizer_param_groups
            self.zoetic_local_grad_list = self.optimizer.zoetic_local_grad_list
            self.zoetic_remote_grad_list = self.optimizer.zoetic_remote_grad_list

            self.zoetic_worker_process = ZoeticProcess(self.local_optimizer_param_groups, self.remote_optimizer_param_groups,
                                                       self.zoetic_local_grad_list, self.zoetic_remote_grad_list,
                                                       self.zoetic_stop_event,self.zoetic_update_flag,self.zoetic_group_no, 
                                                       self.zoetic_remote_lock, self.zoetic_local_lock)
           
            self.zoetic_worker_process.start()
