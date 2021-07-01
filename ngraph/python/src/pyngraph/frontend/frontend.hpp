// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "frontend_manager/frontend_manager.hpp"

namespace py = pybind11;

class FrontEndWrapper {
    std::shared_ptr<ngraph::frontend::FrontEndManager> m_feManager;
    ngraph::frontend::FrontEnd::Ptr m_frontEnd;
public:
    FrontEndWrapper(const std::shared_ptr<ngraph::frontend::FrontEndManager>& feManager,
                      const ngraph::frontend::FrontEnd::Ptr& actual): m_feManager(feManager),
    m_frontEnd(actual){
    }
    const ngraph::frontend::FrontEnd::Ptr& get() const { return m_frontEnd; }
    ngraph::frontend::FrontEnd::Ptr& get() { return m_frontEnd; }
    const std::shared_ptr<ngraph::frontend::FrontEndManager>& getFeManager() const { return m_feManager; }
    std::shared_ptr<ngraph::frontend::FrontEndManager>& getFeManager() { return m_feManager; }
};

void regclass_pyngraph_FrontEnd(py::module m);
