// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include "cast.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs cast (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto out_dtype = node.get_attribute<ngraph::element::Type>("out_dtype");
 
    return node.default_single_output_mapping({std::make_shared<ngraph::opset7::Convert>(data, out_dtype)}, {"Out"});
}

}}}}