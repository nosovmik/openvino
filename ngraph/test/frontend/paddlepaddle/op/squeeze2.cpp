// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using squeezeTestParam = FrontendOpTestParam;
using squeezeTest = FrontendOpTest;

static squeezeTestParam squeeze2() {
    squeezeTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "squeeze2"; //TODO: compact model/decomposited

    //Inputs inputs;
    res.inputs.emplace_back(test::NDArray<float, 4>{{{{0.9197787, 0.17733535, 0.53042966, 0.86230236, 0.36841938 }},
                                                    {{0.3493094, 0.16514555, 0.8016413, 0.13844138, 0.75991917 }},
                                                    {{0.054560415, 0.8210127, 0.6733729, 0.06564956, 0.9250301 }}}}
                            .get_vector());

    res.inputs.emplace_back(test::NDArray<float, 1>{1}
                            .get_vector());

    // 
    res.expected_outputs.emplace_back(test::NDArray<float, 3>({{{0.9197787, 0.17733535, 0.53042966, 0.86230236, 0.36841938 },
                                                                {0.3493094, 0.16514555, 0.8016413, 0.13844138, 0.75991917 },
                                                                {0.054560415, 0.8210127, 0.6733729, 0.06564956, 0.9250301 }}})
                               .get_vector());

    return res;
}

TEST_P(squeezeTest, test_squeeze) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, squeezeTest,
                        ::testing::Values(
                            squeeze2()
                        ),                        
                        squeezeTest::getTestCaseName);                                                 
