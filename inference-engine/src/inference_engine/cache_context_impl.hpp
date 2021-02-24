// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief TODO (mnosov)
 * @file cache_context_impl.hpp
 */

#pragma once

#include <string>

namespace InferenceEngine {

class Core;

class CacheManagerContextImpl {
    const Core& m_core;
    std::string m_deviceName;
    std::string m_modelCacheDir;
public:
    CacheManagerContextImpl(const Core& core, const std::string& deviceName, const std::string& modelCacheDir):
        m_core(core), m_deviceName(deviceName), m_modelCacheDir(modelCacheDir) {
    }
    const Core& getCore() const { return m_core; }
    const std::string& getDeviceName() const { return m_deviceName; }
    const std::string& getModelCacheDir() const { return m_modelCacheDir; }
};

}  // namespace InferenceEngine
