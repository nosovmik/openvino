// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file ie_itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace InferenceEngine {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(IE);
    OV_ITT_DOMAIN(IE_LT);
    OV_ITT_DOMAIN(IE_RT);
}
}
}
