// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <fstream>

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/core_transpose_sinking.h>
#include <ngraph/pass/constant_folding.hpp>

#include "graph.pb.h"

#include "../include/tensorflow_frontend/tensorflow.hpp"

#include "ngraph_builder.h"

using namespace google;

using namespace ngraph::frontend;

InputModelTensorflow::InputModelTensorflow (const std::string& _path) : path(_path)
{
    std::ifstream pb_stream(path, std::ios::binary);
    graph_def = std::make_shared<tensorflow::GraphDef>();
    std::cout << "[ INFO ] Model Parsed: " << graph_def->ParseFromIstream(&pb_stream) << std::endl;
    std::cout << "[ INFO ] Loaded model contains " << graph_def->node_size() << " nodes." << std::endl;
}

std::vector<Place::Ptr> InputModelTensorflow::getInputs () const {
// TODO: Cache results
    std::vector<Place::Ptr> result;
    for (size_t i = 0; i < graph_def->node_size(); ++i) {
        if (graph_def->node(i).op() == "Placeholder")
            result.push_back(std::make_shared<PlaceTensorflow>(graph_def->node(i).name()));
    }
    return result;
}

void InputModelTensorflow::setPartialShape (Place::Ptr place, const ngraph::PartialShape& pshape) {
    auto place_tf = std::dynamic_pointer_cast<PlaceTensorflow>(place);
    partialShapes[place_tf->name] = pshape;
}

std::shared_ptr<ngraph::Function> ngraph::frontend::FrontEndTensorflow::convert (InputModel::Ptr model) const
{
    auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTensorflow>(model);
    std::cerr << "[ INFO ] FrontEndTensorflow::convert invoked\n";

    std::shared_ptr<ngraph::Function> f;
    std::cerr << "[ STATUS ] TranslateGraph return: " << tensorflow::ngraph_bridge::Builder::TranslateGraph(
            model_tf->partialShapes, {}, model_tf->graph_def.get(), "here_should_be_a_graph_name", f) << "\n";
    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;
    std::cerr << "[ STATUS ] Running Transpose Sinking transformation\n";

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::CoreTransposeSinking>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.run_passes(f);

    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;
    return f;
}
