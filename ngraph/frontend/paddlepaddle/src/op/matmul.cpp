// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include "matmul.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

    NamedOutputs matmul(const NodeContext& node) {
        auto x = node.get_ng_input("X");
        auto y = node.get_ng_input("Y");
        auto alpha = node.get_attribute<float>("alpha");
        auto transpose_a = node.get_attribute<bool>("transpose_a");
        auto transpose_b = node.get_attribute<bool>("transpose_b");
        auto mm = std::make_shared<ngraph::opset7::MatMul>(x, y, transpose_a, transpose_b);
        auto alpha_node = ngraph::opset7::Constant::create(ngraph::element::f32, {1}, {alpha});
        return node.default_single_output_mapping({std::make_shared<ngraph::opset7::Multiply>(mm, alpha_node)}, {"Out"});
    }

} // namespace op
} // namespace pdpd
} // namespace frontend
} // namespace ngraph