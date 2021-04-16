// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../shared/include/op.hpp"
// library taken from https://github.com/llohse/libnpy
#include "../shared/include/npy.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

namespace fuzzyOp {
    using PDPDFuzzyOpTest = FrontendOpTest;
    using PDPDFuzzyOpTestParam = FrontendOpTestParam;                     

    static PDPDFuzzyOpTestParam fuzzy_op() {
        PDPDFuzzyOpTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgPool_test1"; // TODO: to read models from config file.
        
        auto modelpath = std::string(TEST_FILES) + res.m_modelsPath + res.m_modelName;

        auto _load_from_npy = [&](std::string name) {
            auto file_path = name + ".npy";

            std::cout << "************load_from_npy (" << file_path << ")" << std::endl;

            std::ifstream npy_file(file_path);
            std::vector<unsigned long> npy_shape;
            std::vector<float> npy_data;
            if (npy_file.good())
                npy::LoadArrayFromNumpy(file_path, npy_shape, npy_data);

            return npy_data;
        };  

        auto npy_input = _load_from_npy(modelpath+"/input0");
        auto npy_output = _load_from_npy(modelpath+"/output0");
        if (npy_input.empty() || npy_output.empty()) {
            throw std::runtime_error("failed to load test case input/output npy file!");
        }

        // TODO: to support more inputs/outputs
        std::vector<float> data_input(npy_input.size());
        std::copy_n(npy_input.data(), npy_input.size(), data_input.begin());

        std::vector<float> data_output(npy_output.size());
        std::copy_n(npy_output.data(), npy_output.size(), data_output.begin()); 

        res.inputs.emplace_back(data_input);

        res.expected_outputs.emplace_back(npy_output);                                

        return res;
    }

    TEST_P(PDPDFuzzyOpTest, test_fuzzy) {
        validateOp();
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest, PDPDFuzzyOpTest,
                            ::testing::Values(
                                fuzzy_op()
                            ),                        
                            PDPDFuzzyOpTest::getTestCaseName);                                                 

}
