// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/batch_norm.hpp"
#include "op/relu.hpp"
#include "op/pool2d.hpp"
#include "op/elementwise_ops.hpp"
#include "op/conv2d.hpp"
#include "op/matmul.hpp"
#include "op/mul.hpp"
#include "op/pool2d.hpp"
#include "op/relu.hpp"
#include "op/reshape2.hpp"
#include "op/scale.hpp"
#include "op/leakyrelu.hpp"
#include "op/interp.hpp"
#include "op/concat.hpp"
#include "op/cast.hpp"
#include "op/softmax.hpp"
#include "op/split.hpp"

#include "op_table.hpp"


namespace ngraph {
namespace frontend {
namespace pdpd {

std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
            {"batch_norm", op::batch_norm},
            {"conv2d", op::conv2d},
            {"elementwise_add", op::elementwise_add},
            {"elementwise_sub", op::elementwise_sub},
            {"elementwise_mul", op::elementwise_mul},
            {"elementwise_div", op::elementwise_div},
            {"elementwise_min", op::elementwise_min},
            {"elementwise_max", op::elementwise_max},
            {"elementwise_pow", op::elementwise_pow},
            {"matmul", op::matmul},
            {"mul", op::mul},
            {"pool2d", op::pool2d},
            {"relu", op::relu},
            {"reshape2", op::reshape2},
            {"scale", op::scale},
            {"leaky_relu", op::leaky_relu},
            {"nearest_interp_v2", op::nearest_interp_v2},
            {"concat", op::concat},
            {"cast", op::cast},
            {"softmax", op::softmax},
            {"split", op::split}
    };
};

}}}
