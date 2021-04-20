//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************


#include <algorithm>
#include <numeric>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#include "framework.pb.h"

#include <paddlepaddle_frontend/model.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset7.hpp>

#include <paddlepaddle_frontend/utility.hpp>
#include "decoder.hpp"
#include "node_context.hpp"
#include "op_table.hpp"

#include <functional>

using namespace ngraph::opset7;

namespace ngraph {
namespace frontend {
namespace pdpd {

NamedOutputs make_ng_node(std::map<std::string, Output<Node>>& nodes,
                          const std::shared_ptr<OpPlacePDPD>& op_place,
                          const std::map<std::string, CreatorFunction>& CREATORS_MAP) {
    const auto& op = op_place->getDesc();
    std::cout << "Making node: " << op->type() << std::endl;

    PDPD_ASSERT(CREATORS_MAP.find(op->type()) != CREATORS_MAP.end(), "No creator found");
    std::map<std::string, OutputVector> named_inputs;
    const auto& input_ports = op_place->getInputPorts();
    for (const auto& name_to_ports : input_ports) {
        for (const auto& port : name_to_ports.second) {
            const auto& var_desc = port->getSourceTensorPDPD()->getDesc();
            if (nodes.count(var_desc->name()))
                named_inputs[name_to_ports.first].push_back(nodes.at(var_desc->name()));
            else
                return NamedOutputs();
        }
    }

    return CREATORS_MAP.at(op->type())(NodeContext(*op, named_inputs));
}

bool endsWith(const std::string &str, const std::string &suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

} // namespace pdpd

std::shared_ptr<Constant> FrontEndPDPD::read_tensor(const std::shared_ptr<TensorPlacePDPD>& tensor_place,
                const std::shared_ptr<InputModelPDPD>& model)
{
    const auto& var_desc = tensor_place->getDesc();
    std::cout << "Reading tensor " << var_desc->name() << std::endl;
    PDPD_ASSERT(var_desc->type().type() == paddle::framework::proto::VarType::LOD_TENSOR);
    const auto& tensor = var_desc->type().lod_tensor().tensor();
    const auto& tensor_length = std::accumulate(
        tensor.dims().cbegin(), tensor.dims().cend(), 1, std::multiplies<int64_t>());
    // TODO: implement for other types
    auto tensor_data = model->readWeight(var_desc->name(), tensor_length);    

    std::vector<size_t> shape(tensor.dims().cbegin(), tensor.dims().cend());
    return Constant::create(element::f32, Shape(shape), tensor_data);
}

std::shared_ptr<Function>
    FrontEndPDPD::convert_model(const std::shared_ptr<InputModelPDPD>& model)
{
    std::cout << "Convert Model Start" << std::endl;    
    
    std::map<std::string, Output<Node>> nodes_dict;
    ParameterVector parameter_nodes;
    ResultVector result_nodes;
    
    const auto& global_var_places = model->getVarPlaces(0);
    for (const auto& name_var : global_var_places)
    {
        const auto& var = name_var.second->getDesc();
        if (pdpd::endsWith(name_var.first, "feed") || pdpd::endsWith(name_var.first, "fetch"))
            continue;
        if (!var->persistable())
            continue;
        nodes_dict[name_var.first] = read_tensor(name_var.second, model);
    }
    std::cout << "Reading consts finished" << std::endl;

    std::map<std::string, pdpd::CreatorFunction> CREATORS_MAP = pdpd::get_supported_ops();
    for (const auto& _inp_place: model->getInputs()) {
        const auto& inp_place = std::dynamic_pointer_cast<TensorPlacePDPD>(_inp_place);
        const auto& var = inp_place->getDesc();
        const auto& shape = inp_place->getPartialShape();
        const auto& type = inp_place->getElementType();
        auto param = std::make_shared<ngraph::opset6::Parameter>(type, shape);
        param->set_friendly_name(var->name());
        nodes_dict[var->name()] = param;
        parameter_nodes.push_back(param);
    }
    
    for (int i = 0; i < model->getBlockNumber(); i++) {
        const auto& op_places = model->getOpPlaces(i);
        for (const auto &op_place : op_places) {
            const auto& op_type = op_place->getDesc()->type();
            std::cerr << "Observing " << op_type << "\n";
            if (op_type == "feed" || op_type == "fetch") {
                // inputs and outputs are stored in the model already
                continue;
            } else {
                const auto& named_outputs = pdpd::make_ng_node(nodes_dict, op_place, CREATORS_MAP);
                // set layer name by the name of first output var
                const auto& first_output_var = op_place->getOutputPorts().begin()->second.at(0)->getTargetTensorPDPD()->getDesc();
                const auto& node = named_outputs.begin()->second[0].get_node_shared_ptr();
                node->set_friendly_name(first_output_var->name());
                std::cerr << "Named with " << node->get_friendly_name() << "\n";

                const auto& out_ports = op_place->getOutputPorts();
                for (const auto& name_to_outputs : named_outputs) {
                    const auto& ports = out_ports.at(name_to_outputs.first);
                    for (size_t idx = 0; idx < ports.size(); ++idx) {
                        const auto& var = ports[idx]->getTargetTensorPDPD()->getDesc();
                        name_to_outputs.second[idx].get_tensor().set_names({var->name()});
                        nodes_dict[var->name()] = name_to_outputs.second[idx];
                    }
                }
            }
        }
    }
    
    for (const auto& _outp_place: model->getOutputs()) {
        const auto& outp_place = std::dynamic_pointer_cast<TensorPlacePDPD>(_outp_place);
        auto var = outp_place->getDesc();
        auto input_var_name = var->name();
        auto result = std::make_shared<ngraph::opset6::Result>(nodes_dict.at(input_var_name));
        result->set_friendly_name(input_var_name + "/Result");
        result_nodes.push_back(result);
    }

    return std::make_shared<ngraph::Function>(result_nodes, parameter_nodes);
}

std::shared_ptr<ngraph::Function> ngraph::frontend::FrontEndPDPD::convert(InputModel::Ptr model) const {
    std::cerr << "[ INFO ] PFrontEndPDPD::convert invoked\n";
    auto pdpd_model = std::dynamic_pointer_cast<ngraph::frontend::InputModelPDPD>(model);    
    auto f = convert_model(pdpd_model);
    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << "\n";
    return f;
}

} // namespace frontend
} // namespace ngraph
