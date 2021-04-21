// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include "scale.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs scale (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto scale = ngraph::opset7::Constant::create(ngraph::element::f32, {1}, {node.get_attribute<float>("scale")});
    return node.default_single_output_mapping({std::make_shared<ngraph::opset7::Multiply>(data, scale)}, {"Out"});
}

}}}}