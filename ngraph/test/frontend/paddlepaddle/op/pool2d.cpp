// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using pool2dTestParam = FrontendOpTestParam;
using pool2dTest = FrontendOpTest;

static pool2dTestParam maxPool() {
    pool2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "maxPool";  //TODO: compact model/decomposited

    //Inputs inputs;
    // data (1, 1, 4, 4) input tensor
    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.0, 2.0, 3.0, 4.0 },
                                                        {5.0, 6.0, 7.0, 8.0 },
                                                        {9.0, 10.0, 11.0, 12.0 },
                                                        {13.0, 14.0, 15.0, 16.0 }}}}}
                            .get_vector());

    // (1, 1, 2, 2)
    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{11.0, 12.0 },
                                                                    {15.0, 16.0 }}}}})
                               .get_vector());

    return res;
}

TEST_P(pool2dTest, test_pool2d) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, pool2dTest,
                        ::testing::Values(
                            maxPool()
                        ),                        
                        pool2dTest::getTestCaseName);                                                 
