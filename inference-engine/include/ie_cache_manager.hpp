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

namespace InferenceEngine {

class Core;

class CacheManagerContextImpl;

/**
 * @brief This class represents Inference Engine Caching Context.
 *
 * Caching context provides information to custom ICacheManager for in/out streams creation
 * CacheManagerContext can't be created by client applications
 */
class INFERENCE_ENGINE_API_CLASS(CacheManagerContext) {
    std::unique_ptr<CacheManagerContextImpl> m_impl;
public:
    /**
     * @brief Constructs CacheManagerContext using internal private implementation
     *
     * @param internalImpl Pointer to internal implementation
     */
    CacheManagerContext(std::unique_ptr<CacheManagerContextImpl>&& internalImpl); // not publicly available
    /** @brief Default destructor
     */
    virtual ~CacheManagerContext();

    /**
     * @brief Reference to Inference Engine Core
     *
     * @return Reference to Inference Engine Core
     */
    const Core& getCore() const;

    /**
     * @brief Reference to device name associated with context
     *
     * @return Reference to device name
     */
    const std::string& getDeviceName() const;

    /**
     * @brief Reference to cache directory
     *
     * @return Reference to cache directory
     */
    const std::string& getModelCacheDir() const;
};

/**
 * @brief This class represents interface for Cache Manager
 *
 * Client can have custom ICacheManager implementation to override default read/write
 * cached models. E.g. use it if you want to encrypt/decrypt cached models
 *
 */
class INFERENCE_ENGINE_API_CLASS(ICacheManager) {
public:
    /**
     * @brief Default destructor
     */
    virtual ~ICacheManager() = default;

    /**
     * @brief Function passing created output stream
     *
     */
    using StreamWriter = std::function<void(std::ostream&)>;
    /**
     * @brief Callback when Inference Engine intends to write network to cache
     *
     * Client needs to call create std::ostream object and call writer(ostream)
     * Otherwise, network will not be cached
     *
     * @param id Id of cache (hash of the network)
     * @param writer Lambda function to be called when stream is created
     * @param cacheManContext Reference to CacheManagerContext
     */
    virtual void writeCacheEntry(const std::string& id, StreamWriter writer,
        const CacheManagerContext& cacheManContext) = 0;

    /**
     * @brief Function passing created input stream
     *
     */
    using StreamReader = std::function<void(std::istream&)>;
    /**
     * @brief Callback when Inference Engine intends to read network from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, network will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the network)
     * @param reader Lambda function to be called when input stream is created
     * @param cacheManContext Reference to CacheManagerContext
     */
    virtual void readCacheEntry(const std::string& id, StreamReader reader,
        const CacheManagerContext& cacheManContext) = 0;

    /**
     * @brief Callback when Inference Engine intends to remove cache entry
     *
     * Client needs to perform appropriate cleanup (e.g. delete a cache file)
     *
     * @param id Id of cache (hash of the network)
     * @param cacheManContext Reference to CacheManagerContext
     */
    virtual void removeCacheEntry(const std::string& id,
        const CacheManagerContext& cacheManContext) = 0;
};

/**
 * @brief File storage-based Implementation of ICacheManager
 *
 * Uses simple file for read/write cached models. No encryption/decryption is used
 *
 */
class FileStorageCacheManager final : public ICacheManager {
    class Impl;
    std::unique_ptr<Impl> m_impl;

public:
    /**
     * @brief Constructor
     *
     */
    explicit FileStorageCacheManager();
    /**
     * @brief Destructor
     *
     */
    ~FileStorageCacheManager() override;

private:
    void writeCacheEntry(const std::string& id, StreamWriter writer,
        const CacheManagerContext& cacheManContext) override;

    void readCacheEntry(const std::string& id, StreamReader reader,
        const CacheManagerContext& cacheManContext) override;

    void removeCacheEntry(const std::string& id,
        const CacheManagerContext& cacheManContext) override;
};

}  // namespace InferenceEngine
