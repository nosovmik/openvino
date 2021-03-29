// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_kernel_selector.h"
#include "one_hot_kernel_ref.h"

namespace kernel_selector {
one_hot_kernel_selector::one_hot_kernel_selector() { Attach<OneHotKernelRef>(); }

KernelsData one_hot_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::ONE_HOT);
}
}  // namespace kernel_selector
