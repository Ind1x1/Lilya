# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CPUOpBuilder


class CPUVertinBuilder(CPUOpBuilder):
    BUILD_VAR = "SN_BUILD_CPU_VERTIN"
    NAME = "cpu_vertin"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.vertin.{self.NAME}_op'

    def sources(self):
        return ['csrc/vertin/cpu_vertin.cpp', 'csrc/vertin/cpu_vertin_impl.cpp']

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return ['csrc/includes']
