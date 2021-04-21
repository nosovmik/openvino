// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

namespace ngraph {
namespace frontend {

inline void PDPD_ASSERT(bool ex, const std::string& msg = "Unspecified error.") {
    if (!ex) throw std::runtime_error(msg);
}

inline void NOT_IMPLEMENTED(const std::string& name = "Unspecified")
{
    throw std::runtime_error(name + " is not implemented");
}

} // namespace frontend
} // namespace ngraph
