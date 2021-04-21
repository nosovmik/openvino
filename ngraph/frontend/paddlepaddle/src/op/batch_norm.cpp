// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include "batch_norm.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs batch_norm (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto gamma = node.get_ng_input("Scale");
    auto beta = node.get_ng_input("Bias");
    auto mean = node.get_ng_input("Mean");
    auto variance = node.get_ng_input("Variance");
    return node.default_single_output_mapping({std::make_shared<ngraph::opset7::BatchNormInference>(
            data, gamma, beta, mean, variance, node.get_attribute<float>("epsilon"))}, {"Y"});
}

}}}}