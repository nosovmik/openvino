// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include "pool2d.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs pool2d (const NodeContext& node) {
    // TODO : resolve padding according to spec
    auto data = node.get_ng_input("X");
    auto pooling_type = node.get_attribute<std::string>("pooling_type");
    auto global_pooling = node.get_attribute<bool>("global_pooling");
    auto adaptive = node.get_attribute<bool>("adaptive");
    auto kernel_shape = node.get_attribute<std::vector<int32_t>>("ksize");
    if (pooling_type == "max" && !global_pooling) {
        auto strides = node.get_attribute<std::vector<int32_t>>("strides");
        auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");
        auto rounding_type = node.get_attribute<bool>("ceil_mode")
                                 ? ngraph::op::RoundingType::CEIL
                                 : ngraph::op::RoundingType::FLOOR;
        return node.default_single_output_mapping({std::make_shared<ngraph::opset7::MaxPool>(
                    data,
                    ngraph::Strides(strides.begin(), strides.end()),
                    ngraph::Shape(paddings.begin(), paddings.end()),
                    ngraph::Shape(paddings.begin(), paddings.end()),
                    ngraph::Shape(kernel_shape.begin(), kernel_shape.end()),
                    rounding_type)}, {"Out"});
    }
    else if (pooling_type == "avg" &&
             (global_pooling || adaptive && all_of(kernel_shape.begin(),
                                                   kernel_shape.end(),
                                                   [](int32_t s) { return s == 1; })))
    {
        // TODO : resolve axes according to rank
        auto axes = ngraph::opset7::Constant::create(ngraph::element::i64, {2}, {2, 3});
        return node.default_single_output_mapping({std::make_shared<ngraph::opset7::ReduceMean>(data, axes, true)}, {"Out"});
    } else {
        throw std::runtime_error("Unsupported pooling type");
    }
}

}}}}