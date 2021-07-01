// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "frontend.hpp"
#include "frontend_manager/frontend_exceptions.hpp"
#include "frontend_manager/frontend_manager.hpp"
#include "pyngraph/function.hpp"

namespace py = pybind11;

void regclass_pyngraph_FrontEnd(py::module m)
{
    py::class_<FrontEndWrapper, std::shared_ptr<FrontEndWrapper>> fe(
        m, "FrontEnd", py::dynamic_attr());
    fe.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    fe.def(
        "load_from_file",
        [](const FrontEndWrapper& frontEnd, const std::string& path) {
            return frontEnd.get()->load_from_file(path);
        },
        py::arg("path"),
        R"(
                Loads an input model by specified model file path.

                Parameters
                ----------
                path : str
                    Main model file path.

                Returns
                ----------
                load_from_file : InputModel
                    Loaded input model.
             )");

    fe.def(
        "convert",
        [](const FrontEndWrapper& frontEnd, ngraph::frontend::InputModel::Ptr model) {
            return frontEnd.get()->convert(model);
        },
        py::arg("model"),
        R"(
                Completely convert and normalize entire function, throws if it is not possible.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert : Function
                    Fully converted nGraph function.
             )");

    fe.def(
        "convert",
        [](const FrontEndWrapper& frontEnd, std::shared_ptr<ngraph::Function> function) {
            return frontEnd.get()->convert(function);
        },
        py::arg("function"),
        R"(
                Completely convert the remaining, not converted part of a function.

                Parameters
                ----------
                function : Function
                    Partially converted nGraph function.

                Returns
                ----------
                convert : Function
                    Fully converted nGraph function.
             )");

    fe.def(
        "convert_partially",
        [](const FrontEndWrapper& frontEnd, ngraph::frontend::InputModel::Ptr model) {
            return frontEnd.get()->convert_partially(model);
        },
        py::arg("model"),
        R"(
                Convert only those parts of the model that can be converted leaving others as-is.
                Converted parts are not normalized by additional transformations; normalize function or
                another form of convert function should be called to finalize the conversion process.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert_partially : Function
                    Partially converted nGraph function.
             )");

    fe.def(
        "decode",
        [](const FrontEndWrapper& frontEnd, ngraph::frontend::InputModel::Ptr model) {
            return frontEnd.get()->decode(model);
        },
        py::arg("model"),
        R"(
                Convert operations with one-to-one mapping with decoding nodes.
                Each decoding node is an nGraph node representing a single FW operation node with
                all attributes represented in FW-independent way.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                decode : Function
                    nGraph function after decoding.
             )");

    fe.def(
        "normalize",
        [](const FrontEndWrapper& frontEnd, std::shared_ptr<ngraph::Function> function) {
            return frontEnd.get()->normalize(function);
        },
        py::arg("function"),
        R"(
                Runs normalization passes on function that was loaded with partial conversion.

                Parameters
                ----------
                function : Function
                    Partially converted nGraph function.
             )");
}
