// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using PDPDFrontendOpTest = FrontendOpTest;

#if 0
static const std::vector<std::string> models {
        std::string("conv2d"),
        std::string("conv2d_s/conv2d.pdmodel"),
        std::string("conv2d_relu/conv2d_relu.pdmodel"),
        std::string("2in_2out/2in_2out.pdmodel"),
};

INSTANTIATE_TEST_CASE_P(PDPDFrontendOpTest, PDPDFrontendOpTest,
                        ::testing::Combine(
                            ::testing::Values(PDPD),
                            ::testing::Values(PATH_TO_MODELS),
                            ::testing::ValuesIn(models)),
                        PDPDFrontendOpTest::getTestCaseName);

#endif

static FrontendOpTestParam getTestData_2in_2out() {
    FrontendOpTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "2in_2out/2in_2out.pdmodel";
    return res;
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, FrontendOpTest,
                        ::testing::Values(
                            getTestData_2in_2out(),
                            FrontendOpTestParam{ PDPD, PATH_TO_MODELS, "conv2d_s/conv2d.pdmodel" },
                            FrontendOpTestParam{ PDPD, PATH_TO_MODELS, "conv2d_relu/conv2d_relu.pdmodel" }
                        ),                        
                        FrontendOpTest::getTestCaseName);

#if 0
static std::vector<FrontendOpTestParam> getTestData_2in_2out() {
    std::vector<FrontendOpTestParam> res;

    res.emplace_back(FrontendOpTestParam{PDPD, PATH_TO_MODELS, "2in_2out/2in_2out.pdmodel"});

    return res;
}

INSTANTIATE_TEST_CASE_P(PDPDFrontendOpTest, PDPDFrontendOpTest,
                        ::testing::ValuesIn(getTestData_2in_2out()),
                        FrontendOpTest::getTestCaseName);  
#endif                                                              