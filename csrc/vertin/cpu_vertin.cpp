// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// Vertin cpu optimizer 

#include "cpu_vertin.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vertin_update" , &sn_vertin_step, "Sonetto Vertin CPU update (C++)");
    m.def("create_vertin" , &create_vertin_optimizer, "Sonetto Vertin CPU update (C++)");
    m.def("destroy_vertin" ,&destroy_vertin_optimizer, "Sonetto Vertin CPU update (C++)");
}

