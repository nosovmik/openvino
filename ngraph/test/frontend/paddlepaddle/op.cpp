// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>

#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "../shared/include/basic_api.hpp"

// library taken from https://github.com/llohse/libnpy
#include "../shared/include/npy.hpp"

using namespace ngraph;
using namespace ngraph::frontend;
using TestEngine = test::IE_CPU_Engine;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

namespace fuzzyOp {
    using PDPDFuzzyOpTest = FrontEndBasicTest; 
    using PDPDFuzzyOpTestParam = std::tuple<std::string,  // FrontEnd name
                                            std::string,  // Base path to models
                                            std::string>; // modelname 

    /*
    // There are 2 versions of PDPD model.
    // * decomposed model, which is a folder.
    // * composed model, which is a file with extension .pdmodel.
    */
    static const std::vector<std::string> models {
            std::string("maxPool_test1/"),
            std::string("avgPool_test1/avgPool_test1.pdmodel"),
    };
                                               
    void run_fuzzy(std::shared_ptr<ngraph::Function> function, std::string& modelfile) {
        auto _load_from_npy = [&](std::string& file_path) {
            std::ifstream npy_file(file_path);
            std::vector<unsigned long> npy_shape;
            std::vector<float> npy_data;
            if (npy_file.good())
                npy::LoadArrayFromNumpy(file_path, npy_shape, npy_data);

            return npy_data;
        };

        auto _get_modelfolder = [&](std::string& modelfile) {
            size_t found = modelfile.find_last_of("/\\");

            return  modelfile.substr(0,found);             
        };

        auto modelfolder = _get_modelfolder(modelfile);

        std::string input_path = modelfolder+"/input0.npy";
        std::string output_path = modelfolder+"/output0.npy";
        auto npy_input = _load_from_npy(input_path);
        auto npy_output = _load_from_npy(output_path);
        if (npy_input.empty() || npy_output.empty()) {
            throw std::runtime_error("failed to load test case input/output npy file!");
        }

        // TODO: to support more inputs/outputs
        std::vector<float> data_input(npy_input.size());
        std::copy_n(npy_input.data(), npy_input.size(), data_input.begin());

        std::vector<float> data_output(npy_output.size());
        std::copy_n(npy_output.data(), npy_output.size(), data_output.begin()); 

        // run test
        auto test_case = test::TestCase<TestEngine>(function);

        test_case.add_input(data_input);
        test_case.add_expected_output(npy_output);
            
        test_case.run();
    }

    TEST_P(PDPDFuzzyOpTest, test_fuzzy) {
        // load
        ASSERT_NO_THROW(doLoadFromFile());

        // convert
        std::shared_ptr<ngraph::Function> function;
        function = m_frontEnd->convert(m_inputModel);
        ASSERT_NE(function, nullptr);

        // run
        run_fuzzy(function, m_modelFile);
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest, PDPDFuzzyOpTest,
                        ::testing::Combine(
                            ::testing::Values(PDPD),
                            ::testing::Values(PATH_TO_MODELS),
                            ::testing::ValuesIn(models)),                 
                            PDPDFuzzyOpTest::getTestCaseName);                                                 

}
