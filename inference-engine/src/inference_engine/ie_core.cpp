// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <istream>
#include <mutex>
#include <sys/stat.h>

#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/graph_util.hpp>

#include <cpp_interfaces/exception2status.hpp>
#include "compilation_context.hpp"
#include "ie_plugin_cpp.hpp"
#include "ie_plugin_config.hpp"
#include "ie_cache_manager.hpp"
#include "ie_itt.hpp"
#include "file_utils.h"
#include "ie_network_reader.hpp"
#include "xml_parse_utils.h"
#include <thread>
#ifdef _WIN32
#include <direct.h>
#define makedir(dir) _mkdir(dir)
#else  // _WIN32
#include <unistd.h>
#define makedir(dir) mkdir(dir, 0755)
#endif  // _WIN32

using namespace InferenceEngine::PluginConfigParams;
using namespace std::placeholders;

namespace InferenceEngine {

namespace {

template <typename T>
struct Parsed {
    std::string _deviceName;
    std::map<std::string, T> _config;
};

template <typename T = Parameter>
Parsed<T> parseDeviceNameIntoConfig(const std::string& deviceName, const std::map<std::string, T>& config = {}) {
    auto config_ = config;
    auto deviceName_ = deviceName;
    if (deviceName_.find("HETERO:") == 0) {
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else if (deviceName_.find("MULTI:") == 0) {
        deviceName_ = "MULTI";
        config_[InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = deviceName.substr(6);
    } else {
        DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
            config_[KEY_DEVICE_ID] = deviceIDLocal;
        }
    }
    return {deviceName_, config_};
}

Parameter copyParameterValue(const Parameter & value) {
    if (value.is<bool>()) {
        return { value.as<bool>() };
    } else if (value.is<int>()) {
        return { value.as<int>() };
    } else if (value.is<unsigned int>()) {
        return { value.as<unsigned int>() };
    } else if (value.is<float>()) {
        return { value.as<float>() };
    } else if (value.is<std::string>()) {
        return { value.as<std::string>() };
    } else if (value.is<std::vector<std::string> >()) {
        return { value.as<std::vector<std::string> >() };
    } else if (value.is<std::vector<int> >()) {
        return { value.as<std::vector<int> >() };
    } else if (value.is<std::vector<float> >()) {
        return { value.as<std::vector<float> >() };
    } else if (value.is<std::vector<unsigned int> >()) {
        return { value.as<std::vector<unsigned int> >() };
    } else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int> >()) {
        return { value.as<std::tuple<unsigned int, unsigned int, unsigned int> >() };
    } else if (value.is<std::tuple<unsigned int, unsigned int> >()) {
        return { value.as<std::tuple<unsigned int, unsigned int> >() };
    }

    return std::move(value);
}

template <typename F>
void allowNotImplemented(F && f) {
    try {
        f();
    } catch (const NotImplemented & ex) { }
}

}  // namespace

DeviceIDParser::DeviceIDParser(const std::string& deviceNameWithID) {
    deviceName = deviceNameWithID;

    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
        deviceName = deviceNameWithID.substr(0, pos);
        deviceID = deviceNameWithID.substr(pos + 1, deviceNameWithID.size());
    }
}

std::string DeviceIDParser::getDeviceID() const {
    return deviceID;
}

std::string DeviceIDParser::getDeviceName() const {
    return deviceName;
}

std::vector<std::string> DeviceIDParser::getHeteroDevices(std::string fallbackDevice) {
    std::vector<std::string> deviceNames;

    std::string cdevice;
    char delimiter = ',';
    size_t pos = 0;

    while ((pos = fallbackDevice.find(delimiter)) != std::string::npos) {
        deviceNames.push_back(fallbackDevice.substr(0, pos));
        fallbackDevice.erase(0, pos + 1);
    }

    if (!fallbackDevice.empty()) deviceNames.push_back(fallbackDevice);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {
    std::vector<std::string> deviceNames;
    auto trim_request_info = [](std::string device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    std::string device;
    char delimiter = ',';
    size_t pos = 0;
    // in addition to the list of devices, every device can have a #requests in the brackets e.g. "CPU(100)"
    // we skip the #requests info here
    while ((pos = devicesList.find(delimiter)) != std::string::npos) {
        auto d = devicesList.substr(0, pos);
        deviceNames.push_back(trim_request_info(d));
        devicesList.erase(0, pos + 1);
    }

    if (!devicesList.empty()) deviceNames.push_back(trim_request_info(devicesList));

    return deviceNames;
}

class Core::Impl : public ICore {
    // Fields are ordered by deletion order
    ITaskExecutor::Ptr _taskExecutor = nullptr;

    mutable std::map<std::string, InferencePlugin> plugins;

    class CoreConfig final {
    public:
        struct CacheConfig {
            std::string _modelCacheDir {};
            std::shared_ptr<ICacheManager> _cacheManager;
        };
        CoreConfig() = default;

        void setAndUpdate(std::map<std::string, std::string> & config) {
            std::lock_guard<std::mutex> lock(_cacheConfigMutex);
            auto it = config.find(CONFIG_KEY(CACHE_DIR));
            if (it != config.end()) {
                _cacheConfig._modelCacheDir = it->second;
                if (!_cacheConfig._modelCacheDir.empty()) {
                    _cacheConfig._cacheManager = std::make_shared<FileStorageCacheManager>(_cacheConfig._modelCacheDir);
                } else {
                    _cacheConfig._cacheManager = nullptr;
                }

                config.erase(it);
                if (!_cacheConfig._modelCacheDir.empty()) {
                    int err = makedir(_cacheConfig._modelCacheDir.c_str());
                    if (err != 0 && errno != EEXIST) {
                        THROW_IE_EXCEPTION << "Couldn't create cache directory! (err=" << strerror(errno) << ")";
                    }
                }
            }
        }

        // Creating thread-safe copy of config including shared_ptr to ICacheManager
        CacheConfig getCacheConfig() const {
            std::lock_guard<std::mutex> lock(_cacheConfigMutex);
            return _cacheConfig;
        }

    private:
        mutable std::mutex _cacheConfigMutex;
        CacheConfig _cacheConfig;
    };

    class CompiledBlobHeader final {
        std::string _ie_version;

    public:
        CompiledBlobHeader() = default;

        explicit CompiledBlobHeader(const std::string & ie_version) :
            _ie_version(ie_version) {
        }

        const std::string & getIeVersion() const {
            return _ie_version;
        }

        friend std::istream & operator >> (std::istream & stream, CompiledBlobHeader & header) {
            std::string xmlStr;
            std::getline(stream, xmlStr);

            std::cerr << "mnosov: Header " << std::endl;
            std::cerr << xmlStr << std::endl;

            pugi::xml_document document;
            pugi::xml_parse_result res = document.load_string(xmlStr.c_str());

            if (res.status != pugi::status_ok) {
                THROW_IE_EXCEPTION_WITH_STATUS(NETWORK_NOT_READ) << "Error reading compiled blob header";
            }

            pugi::xml_node compiledBlobNode = document.document_element();
            header._ie_version = XMLParseUtils::GetStrAttr(compiledBlobNode, "ie_version");

            return stream;
        }

        friend std::ostream & operator << (std::ostream & stream, const CompiledBlobHeader & header) {
            pugi::xml_document document;
            auto compiledBlobNode = document.append_child("compiled_blob");
            compiledBlobNode.append_attribute("ie_version").set_value(header._ie_version.c_str());

            document.save(stream, nullptr, pugi::format_raw);
            document.reset();
            stream << std::endl;

            return stream;
        }
    };

    // Core settings (cache directory, etc)
    CoreConfig coreConfig;

    struct PluginDescriptor {
        FileUtils::FilePath libraryLocation;
        std::map<std::string, std::string> defaultConfig;
        std::vector<FileUtils::FilePath> listOfExtentions;
    };

    std::unordered_set<std::string> opsetNames;
    std::vector<IExtensionPtr> extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;
    mutable std::mutex pluginsMutex;  // to lock parallel access to pluginRegistry and plugins

    bool DeviceSupportsImportExport(const InferencePlugin& plugin) const {
        std::vector<std::string> supportedMetricKeys = plugin.GetMetric(METRIC_KEY(SUPPORTED_METRICS), {});
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
                            METRIC_KEY(IMPORT_EXPORT_SUPPORT));
        bool supported = (it != supportedMetricKeys.end()) &&
                    plugin.GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {});
        return supported;
    }

    ExecutableNetwork LoadNetworkImpl(const CNNNetwork& network,
                                      InferencePlugin& plugin,
                                      const std::map<std::string, std::string>& parsedConfig,
                                      const RemoteContext::Ptr& context,
                                      const std::string& blobID) {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::Impl::LoadNetworkImpl");
        ExecutableNetwork execNetwork;
        execNetwork = context ? plugin.LoadNetwork(network, context, parsedConfig) :
                                plugin.LoadNetwork(network, parsedConfig);
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            try {
                // need to export network for further import from "cache"
                OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::Export");
                std::cerr << "mnosov: start export:[";
                std::cerr << blobID << "]\n";
                cacheManager->writeCacheEntry(blobID, std::bind([&](std::ostream& networkStream) {
                    networkStream << CompiledBlobHeader(GetInferenceEngineVersion()->buildNumber);
                    execNetwork.Export(networkStream);
                }, _1));
            } catch (...) {
                cacheManager->removeCacheEntry(blobID);
                throw;
            }
        }
        return execNetwork;
    }

    ExecutableNetwork LoadNetworkFromCache(std::shared_ptr<ICacheManager> cacheManager,
                                           const std::string& blobId,
                                           InferencePlugin& plugin,
                                           const std::map<std::string, std::string>& config,
                                           const RemoteContext::Ptr& context,
                                           bool& networkIsImported) {
        ExecutableNetwork execNetwork;
        struct HeaderException {};
        std::cerr << "mnosov: LoadNetworkFromCache: " << blobId << "\n";

        IE_ASSERT(cacheManager != nullptr);
        bool headerLoaded = false;
        try {
            cacheManager->readCacheEntry(blobId, std::bind([&](std::istream &networkStream) {
                OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetworkFromCache::ReadStreamAndImport");
                try {
                    CompiledBlobHeader header;
                    networkStream >> header;
                    if (header.getIeVersion() != GetInferenceEngineVersion()->buildNumber) {
                        // network cannot be read
                        throw NetworkNotRead("Version does not match");
                    }
                    headerLoaded = true;
                } catch (...) {
                    throw HeaderException();
                }

                std::cerr << "mnosov: Try import: " << blobId << "\n";
                execNetwork = context ?
                              plugin.ImportNetwork(networkStream, context, config) :
                              plugin.ImportNetwork(networkStream, config);
                networkIsImported = true;
            }, _1));
        } catch (const HeaderException& ex) {
            // For these exceptions just remove old cache and set that import didn't work
            std::cerr << "mnosov: Header exception: " << blobId << "\n";
            cacheManager->removeCacheEntry(blobId);
            networkIsImported = false;
        } catch (...) {
            std::cerr << "mnosov: Other import exception: " << blobId << "\n";
            cacheManager->removeCacheEntry(blobId);
            throw;
        }
        return execNetwork;
    }

    std::map<std::string, std::string> CreateCompileConfig(const InferencePlugin& plugin,
                                                           const std::string& deviceFamily,
                                                           const std::map<std::string, std::string>& origConfig) const {
        std::map<std::string, Parameter> getMetricConfig;
        auto compileConfig = origConfig;

        // 0. remove DEVICE_ID key
        auto deviceIt = compileConfig.find(CONFIG_KEY(DEVICE_ID));
        if (deviceIt != compileConfig.end()) {
            getMetricConfig[deviceIt->first] = deviceIt->second;
            compileConfig.erase(deviceIt);
        }

        // 1. replace it with DEVICE_ARCHITECTURE value
        std::vector<std::string> supportedMetricKeys =
            plugin.GetMetric(METRIC_KEY(SUPPORTED_METRICS), getMetricConfig);
        auto archIt = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
            METRIC_KEY(DEVICE_ARCHITECTURE));
        if (archIt != supportedMetricKeys.end()) {
            auto value = plugin.GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), getMetricConfig);
            compileConfig[METRIC_KEY(DEVICE_ARCHITECTURE)] = value.as<std::string>();
        } else {
            // WA: take device name at least if device does not support DEVICE_ARCHITECTURE metric
            compileConfig[METRIC_KEY(DEVICE_ARCHITECTURE)] = deviceFamily;
        }
        return compileConfig;
    }

    std::string CalculateNetworkHash(const CNNNetwork& network, const std::string& deviceFamily,
                                     const InferencePlugin& plugin,
                                     const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
        std::ostringstream sstr;
        sstr << network.getName();
        sstr << std::this_thread::get_id();
        std::string modelName = sstr.str();
        std::cerr << "mnosov: model name is [" << modelName << "]\n";
        NetworkCompilationContext context(modelName, compileConfig);
        return context.computeHash();
    }

    std::string CalculateFileHash(const std::string& modelName, const std::string& deviceFamily,
                                  const InferencePlugin& plugin,
                                  const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);

        NetworkCompilationContext context(modelName, compileConfig);
        return context.computeHash();
    }

public:
    Impl() {
        opsetNames.insert("opset1");
        opsetNames.insert("opset2");
        opsetNames.insert("opset3");
        opsetNames.insert("opset4");
        opsetNames.insert("opset5");
        opsetNames.insert("opset6");
    }

    ~Impl() override = default;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file. The function supports UNICODE path
     * @param xmlConfigFile An .xml configuraion with device / plugin information
     */
    void RegisterPluginsInRegistry(const std::string& xmlConfigFile) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto parse_result = ParseXml(xmlConfigFile.c_str());
        if (!parse_result.error_msg.empty()) {
            THROW_IE_EXCEPTION << parse_result.error_msg;
        }

        pugi::xml_document& xmlDoc = *parse_result.xml;

        using namespace XMLParseUtils;
        pugi::xml_node ieNode = xmlDoc.document_element();
        pugi::xml_node devicesNode = ieNode.child("plugins");

        FOREACH_CHILD(pluginNode, devicesNode, "plugin") {
            std::string deviceName = GetStrAttr(pluginNode, "name");
            FileUtils::FilePath pluginPath = FileUtils::toFilePath(GetStrAttr(pluginNode, "location").c_str());

            if (deviceName.find('.') != std::string::npos) {
                THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
            }

            // append IR library path for default IE plugins
            {
                FileUtils::FilePath absFilePath = FileUtils::makePath(getInferenceEngineLibraryPath(), pluginPath);
                if (FileUtils::fileExist(absFilePath)) pluginPath = absFilePath;
            }

            // check properties
            auto propertiesNode = pluginNode.child("properties");
            std::map<std::string, std::string> config;

            if (propertiesNode) {
                FOREACH_CHILD(propertyNode, propertiesNode, "property") {
                    std::string key = GetStrAttr(propertyNode, "key");
                    std::string value = GetStrAttr(propertyNode, "value");
                    config[key] = value;
                }
            }

            // check extensions
            auto extensionsNode = pluginNode.child("extensions");
            std::vector<FileUtils::FilePath> listOfExtentions;

            if (extensionsNode) {
                FOREACH_CHILD(extensionNode, extensionsNode, "extension") {
                    FileUtils::FilePath extensionLocation = FileUtils::toFilePath(GetStrAttr(extensionNode, "location").c_str());
                    listOfExtentions.push_back(extensionLocation);
                }
            }

            // fill value in plugin registry for later lazy initialization
            {
                PluginDescriptor desc = {pluginPath, config, listOfExtentions};
                pluginRegistry[deviceName] = desc;
            }
        }
    }

    //
    // ICore public API
    //

    /**
     * @brief Returns global task executor
     * @return Reference to task executor
     */
    ITaskExecutor::Ptr GetTaskExecutor() const override {
        return _taskExecutor;
    }

    CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::Impl::ReadNetwork from file");
        return details::ReadNetwork(modelPath, binPath, extensions);
    }

    CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::Impl::ReadNetwork from memory");
        return details::ReadNetwork(model, weights, extensions);
    }

    ExecutableNetwork LoadNetwork(const CNNNetwork& network, const RemoteContext::Ptr& context,
                                  const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
        if (context == nullptr) {
            THROW_IE_EXCEPTION << "Remote context is nullptr";
        }
        auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        bool loadedFromCache = false;
        ExecutableNetwork res;
        std::string hash;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            hash = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, context, loadedFromCache);
        }

        if (!loadedFromCache) {
            res = LoadNetworkImpl(network, plugin, parsed._config, context, hash);
        }
        return res;
    }

    ExecutableNetwork LoadNetwork(const CNNNetwork& network, const std::string& deviceName,
                                  const std::map<std::string, std::string>& config) override {
        ExecutableNetwork res;
        static int attempt = 0;
        try {
            if (attempt % 3 == 0) {
                OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::CNN0");
                return DoLoadNetwork(network, deviceName, config);
            } else if (attempt % 3 == 1) {
                OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::CNN1");
                return DoLoadNetwork(network, deviceName, config);
            } else if (attempt % 3 == 2) {
                OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::CNN2");
                return DoLoadNetwork(network, deviceName, config);
            }
            attempt++;
        } catch (...) {
            std::cerr << "mnosov: loadNetwork failed" << attempt << "\n";
            throw;
        }
        return res;
    }
    ExecutableNetwork DoLoadNetwork(const CNNNetwork& network, const std::string& deviceName,
                                  const std::map<std::string, std::string>& config) {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::CNN");
        std::cerr << "mnosov: start loading " << network.getName() << "\n";
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        bool loadedFromCache = false;
        ExecutableNetwork res;
        std::string hash;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            hash = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, nullptr, loadedFromCache);
        }

        if (!loadedFromCache) {
            res = LoadNetworkImpl(network, plugin, parsed._config, nullptr, hash);
        }
        return res;
    }

    // TODO: In future, for MULTI plug-in this method can be added to ICore interface
    ExecutableNetwork LoadNetwork(const std::string& modelPath, const std::string& deviceName,
                                  const std::map<std::string, std::string>& config) {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::LoadNetwork::Path");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        bool loadedFromCache = false;
        ExecutableNetwork res;
        std::string hash;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            hash = CalculateFileHash(modelPath, parsed._deviceName, plugin, parsed._config);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, nullptr, loadedFromCache);
        }

        if (!loadedFromCache) {
            auto cnnNetwork = ReadNetwork(modelPath, "");
            res = LoadNetworkImpl(cnnNetwork, plugin, parsed._config, nullptr, hash);
        }
        return res;
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);

        if (parsed._deviceName.empty()) {
            THROW_IE_EXCEPTION << "Device name is empty";
        }

        return GetCPPPluginByName(parsed._deviceName).ImportNetwork(networkModel, parsed._config);
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const RemoteContext::Ptr& context,
                                    const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");

        if (context == nullptr) {
            THROW_IE_EXCEPTION << "Remote context is null";
        }

        auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);
        return GetCPPPluginByName(parsed._deviceName).ImportNetwork(networkModel, context, parsed._config);
    }

    QueryNetworkResult QueryNetwork(const CNNNetwork& network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) const override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::QueryNetwork");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        return GetCPPPluginByName(parsed._deviceName).QueryNetwork(network, parsed._config);
    }

    Parameter GetMetric(const std::string& deviceName, const std::string& name) const override {
        // HETERO case
        {
            if (deviceName.find("HETERO:") == 0) {
                THROW_IE_EXCEPTION
                    << "You can get specific metrics with the GetMetric only for the HETERO itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
            }
        }

        // MULTI case
        {
            if (deviceName.find("MULTI:") == 0) {
                THROW_IE_EXCEPTION
                    << "You can get specific metrics with the GetMetric only for the MULTI itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
            }
        }

        auto parsed = parseDeviceNameIntoConfig(deviceName);

        // we need to return a copy of Parameter object which is created on Core side,
        // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
        // TODO: remove this WA after *-31417 is resolved
        return copyParameterValue(GetCPPPluginByName(parsed._deviceName).GetMetric(name, parsed._config));
    }

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    InferencePlugin GetCPPPluginByName(const std::string& deviceName) const {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "Core::Impl::GetCPPPluginByName");

        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // Plugin is in registry, but not created, let's create

        if (plugins.find(deviceName) == plugins.end()) {
            PluginDescriptor desc = it->second;

            try {
                InferencePlugin plugin(desc.libraryLocation);

                {
                    plugin.SetName(deviceName);

                    // Set Inference Engine class reference to plugins
                    ICore* mutableCore = const_cast<ICore*>(static_cast<const ICore*>(this));
                    plugin.SetCore(mutableCore);
                }

                // Add registered extensions to new plugin
                allowNotImplemented([&](){
                    for (const auto& ext : extensions) {
                        plugin.AddExtension(ext);
                    }
                });

                // configuring
                {
                    allowNotImplemented([&]() {
                        plugin.SetConfig(desc.defaultConfig);
                    });

                    allowNotImplemented([&]() {
                        for (auto&& extensionLocation : desc.listOfExtentions) {
                            plugin.AddExtension(make_so_pointer<IExtension>(extensionLocation));
                        }
                    });
                }

                plugins[deviceName] = plugin;
            } catch (const details::InferenceEngineException& ex) {
                THROW_IE_EXCEPTION << "Failed to create plugin " << FileUtils::fromFilePath(desc.libraryLocation) << " for device " << deviceName
                                   << "\n"
                                   << "Please, check your environment\n"
                                   << ex.what() << "\n";
            }
        }

        return plugins[deviceName];
    }

    /**
     * @brief Unload plugin for specified device, but plugin meta-data is still in plugin registry
     * @param deviceName A name of device
     */
    void UnloadPluginByName(const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);
        auto it = plugins.find(deviceName);
        if (it == plugins.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        plugins.erase(deviceName);
    }

    /**
     * @brief Registers plugin meta-data in registry for specified device
     * @param deviceName A name of device
     */
    void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto it = pluginRegistry.find(deviceName);
        if (it != pluginRegistry.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\"  is already registered in the InferenceEngine";
        }

        if (deviceName.find('.') != std::string::npos) {
            THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
        }

        // append IR library path for default IE plugins
        FileUtils::FilePath pluginPath;
        {
            pluginPath = FileUtils::makePluginLibraryName({}, FileUtils::toFilePath(pluginName.c_str()));

            FileUtils::FilePath absFilePath = FileUtils::makePath(getInferenceEngineLibraryPath(), pluginPath);
            if (FileUtils::fileExist(absFilePath)) pluginPath = absFilePath;
        }

        PluginDescriptor desc = {pluginPath, {}, {}};
        pluginRegistry[deviceName] = desc;
    }

    /**
     * @brief Porvides a list of plugin names in registry; physically such plugins may not be created
     * @return A list of plugin names
     */
    std::vector<std::string> GetListOfDevicesInRegistry() const {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        std::vector<std::string> listOfDevices;
        for (auto&& pluginDesc : pluginRegistry) {
            listOfDevices.push_back(pluginDesc.first);
        }

        return listOfDevices;
    }

    /**
     * @brief Sets config values for a plugin or set of plugins
     * @param deviceName A device name to set config to
     *        If empty, config is set for all the plugins / plugin's meta-data
     * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:FPGA,CPU
     *        just simple forms like CPU, GPU, MULTU, GPU.0, etc
     */
    void SetConfigForPlugins(const std::map<std::string, std::string>& configMap, const std::string& deviceName) {
        auto config = configMap;

        std::lock_guard<std::mutex> lock(pluginsMutex);

        if (deviceName.empty()) {
            coreConfig.setAndUpdate(config);
        }

        // set config for plugins in registry
        bool configIsSet = false;
        for (auto& desc : pluginRegistry) {
            if (deviceName.empty() || deviceName == desc.first) {
                for (auto&& conf : config) {
                    desc.second.defaultConfig[conf.first] = conf.second;
                }
                configIsSet = true;
            }
        }

        if (!configIsSet && !deviceName.empty()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
            if (deviceName.empty() || deviceName == plugin.first) {
                allowNotImplemented([&]() {
                    plugin.second.SetConfig(config);
                });
            }
        }
    }

    /**
     * @brief Registers the extension in a Core object
     *        Such extensions can be used for both CNNNetwork readers and device plugins
     */
    void AddExtension(const IExtensionPtr& extension) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        std::map<std::string, ngraph::OpSet> opsets = extension->getOpSets();
        for (const auto& it : opsets) {
            if (opsetNames.find(it.first) != opsetNames.end())
                THROW_IE_EXCEPTION << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
            opsetNames.insert(it.first);
        }

        // add extensions for already created plugins
        for (auto& plugin : plugins) {
            try {
                plugin.second.AddExtension(extension);
            } catch (...) {}
        }
        extensions.emplace_back(extension);
    }

    /**
     * @brief Provides a list of extensions
     * @return A list of registered extensions
     */
    const std::vector<IExtensionPtr>& GetExtensions() const {
        return extensions;
    }
};

Core::Core(const std::string& xmlConfigFile) {
    _impl = std::make_shared<Impl>();

    std::string xmlConfigFile_ = xmlConfigFile;
    if (xmlConfigFile_.empty()) {
        // register plugins from default plugins.xml config
        FileUtils::FilePath xmlConfigFileDefault = FileUtils::makePath(getInferenceEngineLibraryPath(), FileUtils::toFilePath("plugins.xml"));
        xmlConfigFile_ = FileUtils::fromFilePath(xmlConfigFileDefault);
    }

    RegisterPlugins(xmlConfigFile_);
}

std::map<std::string, Version> Core::GetVersions(const std::string& deviceName) const {
    std::map<std::string, Version> versions;
    std::vector<std::string> deviceNames;

    {
        // for compatibility with samples / demo
        if (deviceName.find("HETERO") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = DeviceIDParser::getHeteroDevices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("HETERO");
        } else if (deviceName.find("MULTI") == 0) {
            auto pos = deviceName.find_first_of(":");
            if (pos != std::string::npos) {
                deviceNames = DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
            }
            deviceNames.push_back("MULTI");
        } else {
            deviceNames.push_back(deviceName);
        }
    }

    for (auto&& deviceName_ : deviceNames) {
        DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.getDeviceName();

        InferenceEngine::InferencePlugin cppPlugin = _impl->GetCPPPluginByName(deviceNameLocal);
        const Version version = cppPlugin.GetVersion();
        versions[deviceNameLocal] = version;
    }

    return versions;
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

CNNNetwork Core::ReadNetwork(const std::wstring& modelPath, const std::wstring& binPath) const {
    return ReadNetwork(FileUtils::wStringtoMBCSstringChar(modelPath),
                       FileUtils::wStringtoMBCSstringChar(binPath));
}

#endif

CNNNetwork Core::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    return _impl->ReadNetwork(modelPath, binPath);
}

CNNNetwork Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const {
    return _impl->ReadNetwork(model, weights);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    return _impl->LoadNetwork(network, deviceName, config);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    return _impl->LoadNetwork(network, context, config);
}

ExecutableNetwork Core::LoadNetwork(const std::string& modelPath, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    return _impl->LoadNetwork(modelPath, deviceName, config);
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName, const ParamMap& params) {
    if (deviceName.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support remote context";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, params);
    return _impl->GetCPPPluginByName(parsed._deviceName).CreateContext(parsed._config);
}

RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName) {
    if (deviceName.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support remote context";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, ParamMap());
    return _impl->GetCPPPluginByName(parsed._deviceName).GetDefaultContext(parsed._config);
}

void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        THROW_IE_EXCEPTION
            << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
        THROW_IE_EXCEPTION
            << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }

    _impl->AddExtension(extension);
}

void Core::AddExtension(const IExtensionPtr& extension) {
    _impl->AddExtension(extension);
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "Core::ImportNetwork");

    // TODO: remove once NotImplemented exception is deprecated and not used
    if (deviceName.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support ImportNetwork";
    }
    if (deviceName.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support ImportNetwork";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    return _impl->GetCPPPluginByName(parsed._deviceName).ImportNetwork(modelFileName, parsed._config);
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    return _impl->ImportNetwork(networkModel, deviceName, config);
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const RemoteContext::Ptr& context,
                                      const std::map<std::string, std::string>& config) {
    return _impl->ImportNetwork(networkModel, context, config);
}

QueryNetworkResult Core::QueryNetwork(const CNNNetwork& network, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    return _impl->QueryNetwork(network, deviceName, config);
}

void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    // HETERO case
    if (deviceName.find("HETERO:") == 0) {
        THROW_IE_EXCEPTION << "SetConfig is supported only for HETERO itself (without devices). "
                                "You can configure the devices with SetConfig before creating the HETERO on top.";
    }

    // MULTI case
    if (deviceName.find("MULTI:") == 0) {
        THROW_IE_EXCEPTION << "SetConfig is supported only for MULTI itself (without devices). "
                                "You can configure the devices with SetConfig before creating the MULTI on top.";
    }

    // GPU.0, FPGA.1 cases
    if (deviceName.find(".") != std::string::npos) {
        THROW_IE_EXCEPTION << "SetConfig is supported only for device family itself (without particular device .#). "
                                "You can pass .# as a particular device instance to QueryNetwork, LoadNetwork, ImportNetwork only";
    }

    if (deviceName.empty()) {
        _impl->SetConfigForPlugins(config, std::string());
    } else {
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        _impl->SetConfigForPlugins(parsed._config, parsed._deviceName);
    }
}

Parameter Core::GetConfig(const std::string& deviceName, const std::string& name) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            THROW_IE_EXCEPTION
                << "You can only GetConfig of the HETERO itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            THROW_IE_EXCEPTION
                << "You can only GetConfig of the MULTI itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the MULTI on top.";
        }
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName);

    // we need to return a copy of Parameter object which is created on Core side,
    // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
    // TODO: remove this WA after *-31417 is resolved
    return copyParameterValue(_impl->GetCPPPluginByName(parsed._deviceName).GetConfig(name, parsed._config));
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name) const {
    return _impl->GetMetric(deviceName, name);
}

std::vector<std::string> Core::GetAvailableDevices() const {
    std::vector<std::string> devices;

    std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : _impl->GetListOfDevicesInRegistry()) {
        std::vector<std::string> devicesIDs;

        try {
            Parameter p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string>>();
        } catch (details::InferenceEngineException&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
            THROW_IE_EXCEPTION << "An exception is thrown while trying to create the " << deviceName
                               << " device and call GetMetric: " << ex.what();
        } catch (...) {
            THROW_IE_EXCEPTION << "Unknown exception is thrown while trying to create the " << deviceName
                               << " device and call GetMetric";
        }

        if (devicesIDs.size() > 1) {
            for (auto&& deviceID : devicesIDs) {
                devices.push_back(deviceName + '.' + deviceID);
            }
        } else if (!devicesIDs.empty()) {
            devices.push_back(deviceName);
        }
    }

    return devices;
}

void Core::RegisterPlugin(const std::string& pluginName, const std::string& deviceName) {
    _impl->RegisterPluginByName(pluginName, deviceName);
}

void Core::RegisterPlugins(const std::string& xmlConfigFile) {
    _impl->RegisterPluginsInRegistry(xmlConfigFile);
}

void Core::UnregisterPlugin(const std::string& deviceName_) {
    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();

    _impl->UnloadPluginByName(deviceName);
}

}  // namespace InferenceEngine
