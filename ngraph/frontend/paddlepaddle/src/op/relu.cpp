// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include "relu.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs relu (const NodeContext& node) {
    return node.default_single_output_mapping({std::make_shared<ngraph::opset7::Relu>(node.get_ng_input("X"))}, {"Out"});
}

}}}}