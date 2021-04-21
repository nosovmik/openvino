// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#include "framework.pb.h"

#include <paddlepaddle_frontend/frontend.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset7.hpp>

namespace ngraph {
namespace frontend {

extern std::map<paddle::framework::proto::VarType_Type, ngraph::element::Type> TYPE_MAP;

// TODO: Inherit from one of the ngraph classes
class AttributeNotFound : public std::exception
{};

class DecoderPDPDProto
{
    paddle::framework::proto::OpDesc op;

public:
    explicit DecoderPDPDProto (const paddle::framework::proto::OpDesc& _op) : op(_op) {}

    std::vector<int32_t> get_ints(const std::string& name, const std::vector<int32_t>& def = {}) const;
    int get_int(const std::string& name, int def = 0) const;
    std::vector<float> get_floats(const std::string& name, const std::vector<float>& def = {}) const;
    float get_float(const std::string& name, float def = 0.) const;
    std::string get_str(const std::string& name, const std::string& def = "") const;
    bool get_bool (const std::string& name, bool def = false) const;

    // TODO: Further populate get_XXX methods on demand
    ngraph::element::Type get_dtype(const std::string& name, ngraph::element::Type def) const;

    std::vector<std::string> get_output_names() const;
};

}
}
