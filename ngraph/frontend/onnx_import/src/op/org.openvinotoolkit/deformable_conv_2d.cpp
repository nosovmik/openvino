// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/deformable_conv_2d.hpp"
#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/deformable_convolution.hpp"
#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        OutputVector op::set_1::deformable_conv_2d(const Node& node)
        {
            const OutputVector& inputs = node.get_ng_inputs();
            const auto strides = convpool::get_strides(node);
            const auto dilations = convpool::get_dilations(node);
            const auto paddings = convpool::get_pads(node);

            const auto group = node.get_attribute_value<int64_t>("group", 1);
            const auto deformable_groups =
                node.get_attribute_value<int64_t>("deformable_groups", 1);
            const auto auto_pad_type = convpool::get_auto_pad(node);

            return {std::make_shared<default_opset::DeformableConvolution>(inputs.at(0),
                                                                           inputs.at(1),
                                                                           inputs.at(2),
                                                                           strides,
                                                                           paddings.first,
                                                                           paddings.second,
                                                                           dilations,
                                                                           auto_pad_type,
                                                                           group,
                                                                           deformable_groups)};
        }
    } // namespace onnx_import
} // namespace ngraph
