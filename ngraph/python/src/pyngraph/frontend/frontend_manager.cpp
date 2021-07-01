// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "frontend.hpp"
#include "frontend_manager.hpp"
#include "frontend_manager/frontend_exceptions.hpp"

namespace py = pybind11;

namespace py = pybind11;
using namespace ngraph::frontend;

void regclass_pyngraph_FrontEndManager(py::module m)
{
    py::class_<FrontEndManagerWrapper, std::shared_ptr<FrontEndManagerWrapper>> fem(
        m, "FrontEndManager", py::dynamic_attr());
    fem.doc() = "ngraph.impl.FrontEndManager wraps ngraph::frontend::FrontEndManager";

    fem.def(py::init<>());

    // Empty pickle dumps are supported as FrontEndManager doesn't have any state
    fem.def(py::pickle([](const FrontEndManagerWrapper&) { return py::make_tuple(0); },
                       [](py::tuple t) { return FrontEndManagerWrapper(); }));

    fem.def(
        "get_available_front_ends",
        [](const FrontEndManagerWrapper& fem) { return fem.get()->get_available_front_ends(); },
        R"(
                Gets list of registered frontends.

                Returns
                ----------
                get_available_front_ends : List[str]
                    List of available frontend names.
             )");
    fem.def(
        "load_by_framework",
        [](const FrontEndManagerWrapper& fem, const std::string& framework, int fec) {
            return FrontEndWrapper(fem.get(), fem.get()->load_by_framework(framework, fec));
        },
        py::arg("framework"),
        py::arg("capabilities") = ngraph::frontend::FrontEndCapabilities::FEC_DEFAULT,
        R"(
                Loads frontend by name of framework and capabilities.

                Parameters
                ----------
                framework : str
                    Framework name. Throws exception if name is not in list of available frontends.

                capabilities : int
                    Frontend capabilities. Default is FrontEndCapabilities.FEC_DEFAULT. It is recommended to use only
                    those capabilities which are needed to minimize load time.

                Returns
                ----------
                load_by_framework : FrontEnd
                    Frontend interface for further loading of models.
             )");
}

void regclass_pyngraph_FEC(py::module m)
{
    class FeCaps
    {
    public:
        int get_caps() const { return m_caps; }

    private:
        int m_caps;
    };

    py::class_<FeCaps, std::shared_ptr<FeCaps>> type(m, "FrontEndCapabilities");
    // type.doc() = "FrontEndCapabilities";
    type.attr("DEFAULT") = ngraph::frontend::FrontEndCapabilities::FEC_DEFAULT;
    type.attr("CUT") = ngraph::frontend::FrontEndCapabilities::FEC_CUT;
    type.attr("NAMES") = ngraph::frontend::FrontEndCapabilities::FEC_NAMES;
    type.attr("WILDCARDS") = ngraph::frontend::FrontEndCapabilities::FEC_WILDCARDS;

    type.def(
        "__eq__",
        [](const FeCaps& a, const FeCaps& b) { return a.get_caps() == b.get_caps(); },
        py::is_operator());
}

void regclass_pyngraph_GeneralFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::GeneralFailure> exc(std::move(m), "GeneralFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::GeneralFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_OpValidationFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::OpValidationFailure> exc(std::move(m),
                                                                    "OpValidationFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::OpValidationFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_OpConversionFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::OpConversionFailure> exc(std::move(m),
                                                                    "OpConversionFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::OpConversionFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_InitializationFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::InitializationFailure> exc(std::move(m),
                                                                      "InitializationFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::InitializationFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_NotImplementedFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::NotImplementedFailure> exc(std::move(m),
                                                                      "NotImplementedFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::NotImplementedFailure& e)
        {
            exc(e.what());
        }
    });
}
