// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

#include <string>
#include <vector>
#include <map>
#include <xml_parse_utils.h>

#include "ie_itt.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "transformations/serialize.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_cnn_network.h"
#include "details/ie_exception.hpp"

#include "ngraph/pass/pass.hpp"
#include "ngraph/variant.hpp"
#include "ngraph/function.hpp"
#include "ngraph/opsets/opset6.hpp"

#ifdef WIN32
#define stat _stat
#endif

namespace InferenceEngine {

struct NetworkCompilationContext final {
    static std::string calculateFileInfo(const std::string& filePath) {
        std::string res;
        struct stat result;
        size_t seed {};
        seed = hash_combine(seed, filePath);
        if (stat(filePath.c_str(), &result) == 0) {
            seed = hash_combine(seed, result.st_mtime);
            seed = hash_combine(seed, result.st_size);
        }
        return std::to_string(seed);
    }

    explicit NetworkCompilationContext(CNNNetwork network,
                                       const std::map<std::string, std::string> & compileOptions = {}) :
            m_weights{},
            m_compileOptions{compileOptions},
            m_inputsInfo{network.getInputsInfo()},
            m_outputsInfo{network.getOutputsInfo()} {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "NetworkCompilationContext::serialize_ir");

        // TODO: mnosov: review this
        try {
            auto & icnnnet = static_cast<ICNNNetwork &>(network);
            auto & ngraphImpl = dynamic_cast<details::CNNNetworkNGraphImpl &>(icnnnet);

            ngraph::pass::Serialize serializer(&m_xmlFile, nullptr,
                ngraph::pass::Serialize::Version::IR_V10, ngraphImpl.getExtensions());
            serializer.run_on_function(ngraphImpl.getFunction());
        } catch (const std::bad_cast &) {
            // IR v7 or older is passed: cannot cast to CNNNetworkNGraphImpl
            m_cachingIsAvailable = false;
            std::cerr << "mnosov: IR v7 is passed; skip import and export" << std::endl;
        } catch (const ngraph::ngraph_error & ex) {
            // failed to serialize the model - caching is not available
            m_cachingIsAvailable = false;
            std::cerr << ex.what() << std::endl;
            std::cerr << "mnosov: failed to serialize the model; skip import and export" << std::endl;
        }

        if (!m_cachingIsAvailable)
            return;

        if (network.getFunction()) {
            // put "affinity", "PrimitivesPriority" runtime information

            for (const auto &op : network.getFunction()->get_ordered_ops()) {
                ngraph::Node::RTMap rt = op->get_rt_info();

                auto affinity_it = rt.find("affinity");
                if (rt.end() != affinity_it) {
                    auto affinity = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(affinity_it->second);
                    m_runtime_attributes += op->get_friendly_name() + "#" + affinity->get();
                }

                auto priorities_it = rt.find("PrimitivesPriority");
                if (rt.end() != priorities_it) {
                    auto primPriority = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(
                            priorities_it->second);
                    m_runtime_attributes += op->get_friendly_name() + "#" + primPriority->get();
                }

                if (const auto &c = std::dynamic_pointer_cast<ngraph::opset6::Constant>(op)) {
                    m_weights.push_back(c);
                }
            }
        }
    }

    explicit NetworkCompilationContext(const std::string& modelName,
                                       const std::map<std::string, std::string>& compileOptions):
                m_compileOptions{compileOptions},
                m_modelName{modelName} {
    }

    bool isCachingAvailable() const {
        return m_cachingIsAvailable;
    }

    std::string computeHash() const {
        IE_ASSERT(isCachingAvailable());
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "NetworkCompilationContext::computeHash");

        size_t seed {};
        if (m_xmlFile.good()) {
            seed = hash_combine(seed, m_xmlFile.str());
        }
        if (!m_modelName.empty()) {
            seed = hash_combine(seed, m_modelName);
        }

        // compute hash on weights if any
        if (!m_weights.empty()) {
            std::cerr << "mnosov: Compute hash on weights" << std::endl;
            for (const auto & c : m_weights) {
                auto data = reinterpret_cast<const std::uint8_t *>(c->get_data_ptr());
                auto data_size = c->get_element_type().size() * ngraph::shape_size(c->get_shape());

                std::uint64_t sum = 0;
                for (size_t i = 0; i < data_size; ++i) {
                    sum += data[i];
                }

                seed = hash_combine(seed, sum);
            }
        }

        for (const auto & kvp : m_compileOptions) {
            seed = hash_combine(seed, kvp.first + kvp.second);
        }

        seed = hash_combine(seed, m_runtime_attributes);

        for (const auto & input : m_inputsInfo) {
            InputInfo::Ptr info = input.second;
            seed = hash_combine(seed, as_int32_t(info->getPrecision()));
            seed = hash_combine(seed, as_int32_t(info->getLayout()));

            const InferenceEngine::PreProcessInfo& preproc = info->getPreProcess();
            // seed = hash_combine(seed, as_int32_t(preproc.getResizeAlgorithm()));
            // seed = hash_combine(seed, as_int32_t(preproc.getColorFormat()));
            seed = hash_combine(seed, as_int32_t(preproc.getMeanVariant()));

            if (preproc.getMeanVariant() == MeanVariant::MEAN_VALUE) {
                seed = hash_combine(seed, preproc.getNumberOfChannels());
                for (size_t c = 0; c < preproc.getNumberOfChannels(); ++c) {
                    const PreProcessChannel::Ptr & channelInfo = preproc[c];
                    seed = hash_combine(seed, channelInfo->stdScale);
                    seed = hash_combine(seed, channelInfo->meanValue);
                }
            } else if (preproc.getMeanVariant() == MeanVariant::MEAN_IMAGE) {
                // TODO: think if we need to compute hash for mean image if it exists
            }
        }

        for (const auto & output : m_outputsInfo) {
            DataPtr info = output.second;
            seed = hash_combine(seed, as_int32_t(info->getPrecision()));
            seed = hash_combine(seed, as_int32_t(info->getLayout()));
        }

        return std::to_string(seed);
    }

private:
    template <typename T>
    static int32_t as_int32_t(T v) {
        return static_cast<int32_t>(v);
    }

    template <typename T>
    static std::size_t hash_combine(std::size_t seed, const T & a) {
        std::size_t val = std::hash<T>()(a);

        // Hash combine formula from boost
        return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

    bool m_cachingIsAvailable = true;

    // network structure (ngraph::Function description)
    using Constant = std::shared_ptr<ngraph::opset6::Constant>;
    std::vector<Constant> m_weights;
    std::stringstream m_xmlFile;

    // compile options
    std::map<std::string, std::string> m_compileOptions;
    std::string                        m_modelName;

    // runtime information
    std::string m_runtime_attributes;
    InferenceEngine::InputsDataMap m_inputsInfo;
    InferenceEngine::OutputsDataMap m_outputsInfo;
};

class CompiledBlobHeader final {
    std::string m_ieVersion;
    std::string m_fileInfo;

public:
    CompiledBlobHeader() = default;

    explicit CompiledBlobHeader(const std::string& ieVersion, const std::string& fileInfo) :
            m_ieVersion(ieVersion),
            m_fileInfo(fileInfo) {
    }

    const std::string& getIeVersion() const {
        return m_ieVersion;
    }

    const std::string& getFileInfo() const {
        return m_fileInfo;
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
        header.m_ieVersion = XMLParseUtils::GetStrAttr(compiledBlobNode, "ie_version");
        header.m_fileInfo = XMLParseUtils::GetStrAttr(compiledBlobNode, "file_info");

        return stream;
    }

    friend std::ostream & operator << (std::ostream & stream, const CompiledBlobHeader & header) {
        pugi::xml_document document;
        auto compiledBlobNode = document.append_child("compiled_blob");
        compiledBlobNode.append_attribute("ie_version").set_value(header.m_ieVersion.c_str());
        compiledBlobNode.append_attribute("file_info").set_value(header.m_fileInfo.c_str());

        document.save(stream, nullptr, pugi::format_raw);
        document.reset();
        stream << std::endl;

        return stream;
    }
};

}  // namespace InferenceEngine
