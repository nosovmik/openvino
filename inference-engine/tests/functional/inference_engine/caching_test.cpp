// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ie_plugin_ptr.hpp"
#include "ngraph/function.hpp"
#include "details/ie_so_loader.h"
#include "ie_metric_helpers.hpp"
#include "ie_iexecutable_network.hpp"

#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_constants.hpp"

#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/network_utils.hpp"

#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;

class CachingInferencePlugin;

class MockRemoteContext : public RemoteContext {
    std::string m_name;
public:
    MockRemoteContext(std::string name): m_name(std::move(name)) {}
    std::string getDeviceName() const noexcept { return m_name; }
    MOCK_METHOD2(CreateBlob, RemoteBlob::Ptr(const TensorDesc&, const ParamMap&));
    MOCK_QUALIFIED_METHOD0(getParams, const, ParamMap());
};

class DummyExecutableNetwork : public ExecutableNetworkInternal {
    CachingInferencePlugin& m_plugin;
public:
    DummyExecutableNetwork(CachingInferencePlugin& plugin) : m_plugin(plugin) {}
    void ExportImpl(std::ostream& networkModel) override;
    IInferRequest::Ptr CreateInferRequest() override {
        return nullptr;
    }
};

class CachingInferencePlugin : public InferenceEngine::InferencePluginInternal {
public:
    std::atomic_int m_loadNetworkCount = {0};
    std::atomic_int m_loadNetworkContextCount = {0};
    std::atomic_int m_importNetworkCount = {0};
    std::atomic_int m_importNetworkContextCount = {0};
    mutable std::atomic_int m_getMetricCount = {0};
    mutable std::atomic_int m_getMetricDevArchCount = {0};
    std::atomic_int m_exportCount = {0};
    std::atomic_int m_getDefaultContextCount = {0};
    bool m_supportImportExport = {true};
    bool m_supportDevArch = {true};
    bool m_throwOnExport = {false};
    bool m_throwOnImport = {false};
    RemoteContext::Ptr m_defContext;

    CachingInferencePlugin(const std::string& name):
        m_defContext(std::make_shared<MockRemoteContext>(name)) {
    }

    ~CachingInferencePlugin() override = default;

    void resetCounters() {
        m_loadNetworkCount = 0;
        m_loadNetworkContextCount = 0;
        m_importNetworkCount = 0;
        m_importNetworkContextCount = 0;
        m_getMetricCount = 0;
        m_exportCount = 0;
        m_getDefaultContextCount = 0;
        m_getMetricDevArchCount = 0;
        m_throwOnExport = false;
        m_throwOnImport = false;
    }

    ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const CNNNetwork& network,
        const std::map<std::string, std::string>& config) override {
        m_loadNetworkCount++;
        return std::make_shared<DummyExecutableNetwork>(*this);
    }

    ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const CNNNetwork& network, RemoteContext::Ptr context,
        const std::map<std::string, std::string>& config) override {
        m_loadNetworkContextCount++;
        if (m_throwOnImport) {
            THROW_IE_EXCEPTION << "Internal test error on import";
        }
        return std::make_shared<DummyExecutableNetwork>(*this);
    }

    ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        m_importNetworkCount++;
        if (m_throwOnImport) {
            THROW_IE_EXCEPTION << "Internal test error on import";
        }
        return ExecutableNetwork(std::make_shared<MockIExecutableNetwork>());
    }

    ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
        const RemoteContext::Ptr& context,
        const std::map<std::string, std::string>& config) override {
        m_importNetworkContextCount++;
        return ExecutableNetwork(std::make_shared<MockIExecutableNetwork>());
    }

    QueryNetworkResult QueryNetwork(const CNNNetwork& network,
                                    const std::map<std::string, std::string>& config) const {
        QueryNetworkResult res;
        auto function = network.getFunction();
        EXPECT_TRUE(function != nullptr);

        for (auto&& node : function->get_ops()) {
            res.supportedLayersMap.emplace(node->get_friendly_name(), m_defContext->getDeviceName());
        }

        return res;
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const override {
        if (METRIC_KEY(SUPPORTED_METRICS) == name) {
            std::vector<std::string> supportedMetrics = {
                METRIC_KEY(IMPORT_EXPORT_SUPPORT)
            };
            if (m_supportDevArch) {
                supportedMetrics.push_back(METRIC_KEY(DEVICE_ARCHITECTURE));
            }
            return supportedMetrics;
        } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
            return std::vector<std::string>();
        } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
            m_getMetricCount++;
            return m_supportImportExport;
        } else if (METRIC_KEY(DEVICE_ARCHITECTURE) == name) {
            if (options.count("DEVICE_ID")) {
                m_getMetricDevArchCount++;
                auto id = options.at("DEVICE_ID").as<std::string>();
                if (std::stoi(id) < 10) {
                    return "mock_first_architecture";
                } else {
                    return "mock_another_architecture";
                }
            }
            return name;
        } else {
            THROW_IE_EXCEPTION << "Unsupported device metric: " << name;
        }
    }

    RemoteContext::Ptr GetDefaultContext(const ParamMap& params) override {
        m_getDefaultContextCount++;
        return m_defContext;
    }
};

void DummyExecutableNetwork::ExportImpl(std::ostream& networkModel) {
    networkModel << "Export is called";
    m_plugin.m_exportCount++;
    if (m_plugin.m_throwOnExport) {
        THROW_IE_EXCEPTION << "Internal test error on export";
    }
}

//------------------------------------------------------
class MkDirGuard {
    std::string m_dir;
public:
    MkDirGuard(const std::string &dir = std::string()): m_dir(dir) {
        if (!m_dir.empty()) {
            CommonTestUtils::makeDir(m_dir);
        }
    }
    ~MkDirGuard() {
        if (!m_dir.empty()) {
            CommonTestUtils::removeFilesWithExt(m_dir, "blob");
            CommonTestUtils::removeDir(m_dir);
        }
    }
};

class CachingTest : public ::testing::Test {
public:
    unique_ptr<SharedObjectLoader> sharedObjectLoader;
    std::function<void(IInferencePlugin*)> injectProxyEngine;
    std::shared_ptr<CachingInferencePlugin> m_plugin; // must be shared_ptr due to internal InferencePluginInternal logic
    std::string modelName = "Caching_test.xml";
    std::string weightsName = "Caching_test.bin";
    std::string deviceName = "mock";
    InferenceEngine::Core ie;
    MkDirGuard m_dirCreator;

    std::string get_mock_engine_name() {
        std::string mockEngineName("mock_engine");
        return CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    }

    void SetUp() override {
        m_dirCreator = MkDirGuard("testCache");
        m_plugin = std::make_shared<CachingInferencePlugin>(deviceName);
        std::string libraryName = get_mock_engine_name();
        sharedObjectLoader.reset(new SharedObjectLoader(libraryName.c_str()));
        injectProxyEngine = make_std_function<void(IInferencePlugin*)>("InjectProxyEngine");
        injectProxyEngine(m_plugin.get());

        FuncTestUtils::TestModel::generateTestModel(modelName, weightsName);

        ie.RegisterPlugin(std::string("mock_engine") + IE_BUILD_POSTFIX, deviceName);
        // TODO: workaround. Performs further load of plugin, otherwise UnregisterPlugin throws exception
        ie.GetDefaultContext(deviceName);
        m_plugin->resetCounters();
    }

    void TearDown() override {
        ie.UnregisterPlugin(deviceName);

        CommonTestUtils::removeIRFiles(modelName, weightsName);
        // remove all cache entries
        m_dirCreator = MkDirGuard();

        m_plugin = nullptr;
    }

    void enableCacheConfig() {
        ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "testCache"} });
    }

    void performLoadByName() {
        auto exeNet = ie.LoadNetwork(modelName, deviceName);
        (void)exeNet;
    }

    void performReadAndLoad() {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        auto exeNet = ie.LoadNetwork(cnnNetwork, deviceName);
        (void)exeNet;
    }

private:
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function <T> ptr(reinterpret_cast<T*>(sharedObjectLoader->get_symbol(functionName.c_str())));
        return ptr;
    }
};

TEST_F(CachingTest, TestReadLoad) {
    enableCacheConfig();

    { // Step 1: read and load network without cache
        performReadAndLoad();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, but now cache must be available from Step 1
        performReadAndLoad();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called instead of load + export'
    }
}

TEST_F(CachingTest, TestLoadByName) {
    enableCacheConfig();

    { // Step 1: read and load network without cache
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, but now cache must be available from Step 1
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called instead of load + export'
    }
}

TEST_F(CachingTest, TestLoadByName_NoCacheSupported) {
    enableCacheConfig();                     // Enable caching in global settings
    m_plugin->m_supportImportExport = false; // but it is not supported by plugin

    { // read and load network without cache
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }
}

TEST_F(CachingTest, TestLoadByName_NoCacheEnabled) {
    { // read and load network without cache
        performLoadByName();

        EXPECT_EQ(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was not called when global cache is disabled'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }
}

TEST_F(CachingTest, TestReadLoadContext) {
    enableCacheConfig();
    auto performReadLoadContext = [&] {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        auto context = ie.GetDefaultContext(deviceName);
        EXPECT_EQ(m_plugin->m_getDefaultContextCount, 1); // verify: 'getDefaultContext' is called
        auto exeNet = ie.LoadNetwork(cnnNetwork, context);
        (void)exeNet;
    };

    { // Step 1: read and load network without cache
        performReadLoadContext();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkContextCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkContextCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, but now cache must be available from Step 1
        performReadLoadContext();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkContextCount, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkContextCount, 1); // verify: 'import was called instead of load + export'
    }
}

TEST_F(CachingTest, TestDeviceArchitecture) {
    enableCacheConfig();
    auto performLoadByNameArch = [&] (const std::string& suffix) {
        auto exeNet = ie.LoadNetwork(modelName, deviceName + "." + suffix);
        (void)exeNet;
    };

    { // Step 1: read and load network without cache
        performLoadByNameArch("0"); // loading "mock.0" device

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
        EXPECT_GT(m_plugin->m_getMetricDevArchCount, 0); // verify: 'getMetric for device architecture was called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, but for device 1. Cache must be reused from Step 1
        performLoadByNameArch("1"); // loading "mock.1" device

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called instead of load + export'
        EXPECT_GT(m_plugin->m_getMetricDevArchCount, 0); // verify: 'getMetric for device architecture was called'
    }

    m_plugin->resetCounters();

    { // Step 3: same load, but for device 50. It has different architecture (see CachingInferencePlugin::GetMetric), so cache will not be reused
        performLoadByNameArch("50"); // loading "mock.50" device

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
        EXPECT_GT(m_plugin->m_getMetricDevArchCount, 0); // verify: 'getMetric for device architecture was called'
    }

    m_plugin->resetCounters();

    { // Step 4: same load, but for device 51. It has different same architecture as #50, so cache will be reused
        performLoadByNameArch("51"); // loading "mock.51" device

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called instead of load + export'
        EXPECT_GT(m_plugin->m_getMetricDevArchCount, 0); // verify: 'getMetric for device architecture was called'
    }
}

TEST_F(CachingTest, TestNoDeviceArchitecture) {
    enableCacheConfig();
    m_plugin->m_supportDevArch = false;
    auto performLoadByNameArch = [&] (const std::string& suffix) {
        auto exeNet = ie.LoadNetwork(modelName, deviceName + "." + suffix);
        (void)exeNet;
    };

    { // Step 1: read and load network without cache
        performLoadByNameArch("0"); // loading "mock.0" device

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
        EXPECT_EQ(m_plugin->m_getMetricDevArchCount, 0); // verify: 'getMetric for device architecture was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, but for device 50. Cache must be reused from Step 1 as arch is "mock" by default
        performLoadByNameArch("50"); // loading "mock.50" device

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 0); // verify: 'load was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called' (optimization works)
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called instead of load + export'
        EXPECT_EQ(m_plugin->m_getMetricDevArchCount, 0); // verify: 'getMetric for device architecture was not called'
    }
}

TEST_F(CachingTest, TestChangeCacheDir) {
    // Dynamic change of cache dir is not supported
    ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "testCache"} });

    { // Step 1: read and load network without cache
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();
    ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "testCache2"} });

    { // Step 2: same load, and due to cache dir change - it shall load network once again without import
        MkDirGuard guard("testCache2"); // remove dir after test
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }
}

TEST_F(CachingTest, TestClearCacheDir) {
    // Once enabled, disabling it not supported anymore
    ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "testCache"} });
    ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), ""} });

    for (int i = 0; i < 3; i++) { // read and load network, no caching
        m_plugin->resetCounters();
        performLoadByName();

        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }
}

TEST_F(CachingTest, TestChangeOtherConfig) {
    // Once enabled, user can set other global configs
    enableCacheConfig();
    ie.SetConfig({ {"otherKey", "otherValue"} });
    { // Step 1: read and load network without cache
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, verify cache is available
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 0); // verify: 'load was not called'
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called'
    }
}

TEST_F(CachingTest, ErrorHandling_throwOnExport) {
    enableCacheConfig();

    { // Step 1: read and load network without cache
        m_plugin->m_throwOnExport = true;
        // TODO: discuss if it shall throw exception on 'export' failure?
        performLoadByName(); // TODO: put EXPECT_THROW....

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, and due to unsuccessful export - it shall load network once again without import
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }
}

TEST_F(CachingTest, ErrorHandling_throwOnImport) {
    enableCacheConfig();
    auto performLoadByName = [&] {
        auto exeNet = ie.LoadNetwork(modelName, deviceName);
        (void)exeNet;
    };

    { // Step 1: read and load network without cache
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    // TODO: discuss expectations for steps 2 and 3
    { // Step 2: same load, import is unsuccessful
        m_plugin->m_throwOnImport = true;
        m_plugin->m_throwOnExport = true; // and simluate unsuccessful export as well
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called'
    }

    m_plugin->resetCounters();

    { // Step 3: same load, cache should be deleted due to unsuccessful import on step 2
        performLoadByName();

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called here'
    }
}

TEST_F(CachingTest, LoadHeteroWithCorrectConfig) {
    enableCacheConfig();

    // TODO: test also HETERO with 1 plugin but different architectures, e.g. "HETERO:mock.1,mock.51"
    std::string heteroDevice = CommonTestUtils::DEVICE_HETERO + std::string(":mock.1,mock.2");
    { // Step 1: read and load network without cache
        ie.LoadNetwork(modelName, heteroDevice);

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 1); // verify: 'load was called'
        EXPECT_EQ(m_plugin->m_exportCount, 1); // verify: 'export was called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 0); // verify: 'import was not called'
    }

    m_plugin->resetCounters();

    { // Step 2: same load, expected import
        ie.LoadNetwork(modelName, heteroDevice);

        EXPECT_GT(m_plugin->m_getMetricCount, 0); // verify: 'getMetric was called'
        EXPECT_EQ(m_plugin->m_loadNetworkCount, 0); // verify: 'load was not called'
        EXPECT_EQ(m_plugin->m_exportCount, 0); // verify: 'export was not called'
        EXPECT_EQ(m_plugin->m_importNetworkCount, 1); // verify: 'import was called'
    }
}
