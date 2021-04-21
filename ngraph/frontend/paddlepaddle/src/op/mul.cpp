// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include <paddlepaddle_frontend/utility.hpp>
#include "mul.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs mul (const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto y = node.get_ng_input("Y");
    PDPD_ASSERT(x.get_partial_shape().rank().is_static(), "matmul: X rank must be static!");
    int64_t x_rank = x.get_partial_shape().rank().get_length();
    PDPD_ASSERT(y.get_partial_shape().rank().is_static() &&
              y.get_partial_shape().rank().get_length() == 2, "matmul: Y rank must be static, and 2!");
    if (x_rank > 2) {
        auto shape = std::make_shared<ngraph::opset7::ShapeOf>(x);
        int64_t x_num_col_dims = node.get_attribute<int32_t>("x_num_col_dims");
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i64, {}, {0});
        auto split_lengths = ngraph::opset7::Constant::create(ngraph::element::i64, {2},
                                                              {x_num_col_dims, x_rank - x_num_col_dims});
        auto split = std::make_shared<ngraph::opset7::VariadicSplit>(shape, axis, split_lengths);
        auto f_dim_red_axis = ngraph::opset7::Constant::create(ngraph::element::i64, {}, {0});
        auto first_dim_reduce = std::make_shared<ngraph::opset7::ReduceProd>(split->output(0),
                                                                             f_dim_red_axis);
        auto f_dim_shape = ngraph::opset7::Constant::create(ngraph::element::i64, {1}, {1});
        auto first_dim = std::make_shared<ngraph::opset7::Reshape>(first_dim_reduce, f_dim_shape, false);
        auto s_dim_red_axis = ngraph::opset7::Constant::create(ngraph::element::i64, {}, {0});
        auto second_dim_reduce = std::make_shared<ngraph::opset7::ReduceProd>(split->output(1),
                                                                              s_dim_red_axis);
        auto s_dim_shape = ngraph::opset7::Constant::create(ngraph::element::i64, {1}, {1});
        auto second_dim = std::make_shared<ngraph::opset7::Reshape>(second_dim_reduce, s_dim_shape, false);
        auto out_shape = std::make_shared<ngraph::opset7::Concat>(ngraph::NodeVector{first_dim, second_dim},
                                                                  0);
        auto x_reshaped = std::make_shared<ngraph::opset7::Reshape>(x, out_shape, false);
        return node.default_single_output_mapping({std::make_shared<ngraph::opset7::MatMul>(x_reshaped, y)}, {"Out"});
    }
    return node.default_single_output_mapping({std::make_shared<ngraph::opset7::MatMul>(x, y)}, {"Out"});

}

}}}}