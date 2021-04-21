// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "concat.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs concat (const NodeContext& node) {
    auto data = node.get_ng_inputs("X");
    auto axis = node.get_attribute<int>("axis");
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Concat>(data, axis)}, {"Out"});
}

}}}}