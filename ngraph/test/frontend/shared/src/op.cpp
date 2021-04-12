// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>
#include "../include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

std::string FrontendOpTest::getTestCaseName(const testing::TestParamInfo<FrontendOpTestParam> &obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    //res += "I" + joinStrings(obj.param.m_oldInputs) + joinStrings(obj.param.m_newInputs);
    //res += "O" + joinStrings(obj.param.m_oldOutputs) + joinStrings(obj.param.m_newOutputs);
    // need to replace special characters to create valid test case name
    res = std::regex_replace(res, std::regex("[/\\.]"), "_");
    return res;
}

void FrontendOpTest::SetUp() {
    initParamTest();
}

void FrontendOpTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = std::string(TEST_FILES) + m_param.m_modelsPath + m_param.m_modelName;
    std::cout << "Model: " << m_param.m_modelName << std::endl;
}

void FrontendOpTest::doLoadFromFile() {
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.availableFrontEnds());
    ASSERT_NO_THROW(m_frontEnd = m_fem.loadByFramework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->loadFromFile(m_param.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);
}

/*---------------------------------------------------------------------------------------------------------------------*/

using TestEngine = test::IE_CPU_Engine;

TEST_P(FrontendOpTest, test_model_runtime) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
//    std::cout << "Ordered ops names\n";
//    for (const auto &n : function->get_ordered_ops()) {
//        std::cout << "----" << n->get_friendly_name() << "---\n";
//    }

    Inputs inputs;
    // data (1, 1, 7, 5) input tensor
    inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                   {5.f, 6.f, 7.f, 8.f, 9.f},
                                                   {10.f, 11.f, 12.f, 13.f, 14.f},
                                                   {15.f, 16.f, 17.f, 18.f, 19.f},
                                                   {20.f, 21.f, 22.f, 23.f, 24.f},
                                                   {25.f, 26.f, 27.f, 28.f, 29.f},
                                                   {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                            .get_vector());

    // filters (1, 1, 3, 3) aka convolution weights
    inputs.emplace_back(
        test::NDArray<float, 4>{{{{{1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}}}}}
            .get_vector());

    // (1, 1, 4, 3)
    auto expected_output = test::NDArray<float, 4>({{{{12.f, 27.f, 24.f},
                                                      {63.f, 108.f, 81.f},
                                                      {123.f, 198.f, 141.f},
                                                      {112.f, 177.f, 124.f}}}})
                               .get_vector();

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}
