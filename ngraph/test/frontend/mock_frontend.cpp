// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"
#include "frontend_manager/frontend_manager_defs.hpp"
#include "frontend_manager/frontend_manager.hpp"

#ifdef mock1_frontend_EXPORTS // defined if we are building the plugin DLL (instead of using it)
#define MOCK_API NGRAPH_HELPER_DLL_EXPORT
#else
#define MOCK_API NGRAPH_HELPER_DLL_IMPORT
#endif // mock1_frontend_EXPORTS

using namespace ngraph;
using namespace ngraph::frontend;

class FrontEndMock: public FrontEnd
{
};

extern "C" MOCK_API char* GetAPIVersion()
{
    return OV_FRONTEND_API_VERSION;
}

extern "C" MOCK_API void* GetFrontEndData()
{
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "mock1";
    res->m_creator = [](ngraph::frontend::FrontEndCapabilities)
            {
                return std::make_shared<FrontEndMock>();
            };
    return res;
}