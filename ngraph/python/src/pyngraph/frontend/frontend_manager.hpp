// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include "frontend_manager/frontend_manager.hpp"

namespace py = pybind11;


class FrontEndManagerWrapper {
    std::shared_ptr<ngraph::frontend::FrontEndManager> m_feManager;
public:
    FrontEndManagerWrapper(): m_feManager(std::make_shared<ngraph::frontend::FrontEndManager>()) {
    }
    const std::shared_ptr<ngraph::frontend::FrontEndManager>& get() const { return m_feManager; }
    std::shared_ptr<ngraph::frontend::FrontEndManager>& get() { return m_feManager; }
};

void regclass_pyngraph_FrontEndManager(py::module m);
void regclass_pyngraph_FEC(py::module m);
void regclass_pyngraph_NotImplementedFailureFrontEnd(py::module m);
void regclass_pyngraph_InitializationFailureFrontEnd(py::module m);
void regclass_pyngraph_OpConversionFailureFrontEnd(py::module m);
void regclass_pyngraph_OpValidationFailureFrontEnd(py::module m);
void regclass_pyngraph_GeneralFailureFrontEnd(py::module m);
