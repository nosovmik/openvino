// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "conv2d.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs conv2d (const NodeContext& node) {
    auto data = node.get_ng_input("Input");
    auto filter = node.get_ng_input("Filter");
    // TODO: resolve padding according to spec
    auto strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");
    auto dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Convolution>(
        data,
        filter,
        ngraph::Strides(strides.begin(), strides.end()),
        ngraph::CoordinateDiff(paddings.begin(), paddings.end()),
        ngraph::CoordinateDiff(paddings.begin(), paddings.end()),
        ngraph::Strides(dilations.begin(), dilations.end()))}, {"Output"});
}

}}}}