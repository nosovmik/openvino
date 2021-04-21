// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include "model.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph {
namespace frontend {

class NGRAPH_API FrontEndPDPD : public FrontEnd
{
    static std::shared_ptr<Function> convert_model(const std::shared_ptr<InputModelPDPD>& model);
    static std::shared_ptr<opset6::Constant> read_tensor(const std::shared_ptr<TensorPlacePDPD>& place,
                                                  const std::shared_ptr<InputModelPDPD>& model);
public:

    FrontEndPDPD ()
    {
    }

    virtual InputModel::Ptr loadFromFile (const std::string& path) const override
    {
        return std::make_shared<InputModelPDPD>(path);
    }

    virtual std::shared_ptr<Function> convert (InputModel::Ptr model) const override;
};

} // namespace frontend
} // namespace ngraph
