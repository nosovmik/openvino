// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine Cache Manager class C++ API
 *
 * @file ie_cache_manager.hpp
 */
#pragma once

#include <memory>
#include <istream>
#include <string>
#include <functional>
#include "ie_api.h"

// TODO: mnosov: Documentation
namespace InferenceEngine {

class Core;

class CacheManagerContextImpl;
class INFERENCE_ENGINE_API_CLASS(CacheManagerContext) {
    std::unique_ptr<CacheManagerContextImpl> m_impl;
public:
    CacheManagerContext(std::unique_ptr<CacheManagerContextImpl>&& internalImpl); // not publicly available
    virtual ~CacheManagerContext();

    const Core& getCore() const;
    const std::string& getDeviceName() const;
    const std::string& getModelCacheDir() const;
};

class INFERENCE_ENGINE_API_CLASS(ICacheManager) {
public:
    virtual ~ICacheManager() = default;

    using StreamWriter = std::function<void(std::ostream&)>;
    virtual void writeCacheEntry(const std::string& id, StreamWriter writer,
        const CacheManagerContext& cacheManContext) = 0;

    using StreamReader = std::function<void(std::istream&)>;
    virtual void readCacheEntry(const std::string& id, StreamReader reader,
        const CacheManagerContext& cacheManContext) = 0;

    virtual void removeCacheEntry(const std::string & id,
        const CacheManagerContext& cacheManContext) = 0;
};

class FileStorageCacheManager final : public ICacheManager {
    class Impl;
    std::unique_ptr<Impl> m_impl;

public:
    explicit FileStorageCacheManager();
    ~FileStorageCacheManager() override;

    void writeCacheEntry(const std::string& id, StreamWriter writer,
        const CacheManagerContext& cacheManContext) override;

    void readCacheEntry(const std::string& id, StreamReader reader,
        const CacheManagerContext& cacheManContext) override;

    void removeCacheEntry(const std::string& id,
        const CacheManagerContext& cacheManContext) override;
};

}  // namespace InferenceEngine
