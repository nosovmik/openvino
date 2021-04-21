// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include "model.hpp"
#include <ngraph/opsets/opset7.hpp>

namespace ngraph {
namespace frontend {

class NGRAPH_API FrontEndPDPD : public FrontEnd
{
public:

    FrontEndPDPD () = default;

    InputModel::Ptr loadFromFile (const std::string& path) const override
    {
        return std::make_shared<InputModelPDPD>(path);
    }

    std::shared_ptr<Function> convert (InputModel::Ptr model) const override;

private:
    static std::shared_ptr<Function> convert_model(const std::shared_ptr<InputModelPDPD>& model);
    static std::shared_ptr<opset7::Constant> read_tensor(const std::shared_ptr<TensorPlacePDPD>& place,
                                                         const std::shared_ptr<InputModelPDPD>& model);

    static void createConstants(const std::shared_ptr<InputModelPDPD>& model,
                                std::map<std::string, Output<Node>>& node_dict);

    static ParameterVector createParameters(const std::shared_ptr<InputModelPDPD>& model,
                                            std::map<std::string, Output<Node>>& node_dict);

    static void createOperations(const std::shared_ptr<InputModelPDPD>& model,
                                 std::map<std::string, Output<Node>>& node_dict);

    static ResultVector createResults(const std::shared_ptr<InputModelPDPD>& model,
                                      std::map<std::string, Output<Node>>& node_dict);
};

} // namespace frontend
} // namespace ngraph
