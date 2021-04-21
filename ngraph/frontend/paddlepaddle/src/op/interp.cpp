// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include "interp.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs nearest_interp_v2 (const NodeContext& node) {
    auto x = node.get_ng_input("X");

    using InterpolateMode = ngraph::opset7::Interpolate::InterpolateMode;
    using CoordinateTransformMode = ngraph::opset7::Interpolate::CoordinateTransformMode;
    using Nearest_mode = ngraph::opset7::Interpolate::NearestMode;
    using InterpolateAttrs = ngraph::opset7::Interpolate::InterpolateAttrs;
    using ShapeCalcMode = ngraph::opset7::Interpolate::ShapeCalcMode;

    InterpolateAttrs attrs;

    attrs.mode = InterpolateMode::nearest; //HARDCODE

    auto out_w = node.get_attribute<int>("out_w");
    auto out_h = node.get_attribute<int>("out_h");
    auto scale = node.get_attribute<std::vector<float>>("scale");
    if (out_w <= 0 || out_h <= 0) {
        attrs.shape_calculation_mode = ShapeCalcMode::scales;
    }
    else {
        attrs.shape_calculation_mode = ShapeCalcMode::sizes;
    }

    auto target_spatial_shape =
        ngraph::opset7::Constant::create<int64_t>(element::i64, Shape{2}, {out_h, out_w});
    auto scales = ngraph::opset7::Constant::create<float>(element::f32, Shape{2}, std::vector<float>(scale.begin(), scale.end()));
    auto axes = ngraph::opset7::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});
    
    attrs.coordinate_transformation_mode = CoordinateTransformMode::asymmetric; //HARDCODE
    attrs.nearest_mode = Nearest_mode::round_prefer_floor; //HARDCODE
    attrs.antialias = false;  //HARDCODE
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    return node.default_single_output_mapping({std::make_shared<ngraph::opset7::Interpolate>(x, target_spatial_shape, scales, axes, attrs)}, {"Out"});
}

}}}}