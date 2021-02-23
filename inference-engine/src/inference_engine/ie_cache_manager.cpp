// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include "ie_cache_manager.hpp"
#include "cache_context_impl.hpp"
#include "file_utils.h"


namespace InferenceEngine {

class FileStorageCacheManager::Impl {
    std::string getBlobFile(const std::string& blobHash, const CacheManagerContext& cacheManContext) const {
        return FileUtils::makePath(cacheManContext.getModelCacheDir(), blobHash + ".blob");
    }

public:
    void writeCacheEntry(const std::string& blobHash, ICacheManager::StreamWriter writer,
        const CacheManagerContext& cacheManContext) {
        writer(std::ofstream(getBlobFile(blobHash, cacheManContext), std::ios_base::binary));
    }

    void removeCacheEntry(const std::string& blobHash,
        const CacheManagerContext& cacheManContext) {
        auto blobFileName = getBlobFile(blobHash, cacheManContext);
        if (FileUtils::fileExist(blobFileName))
            std::remove(blobFileName.c_str());
    }

    void readCacheEntry(const std::string& blobHash, FileStorageCacheManager::StreamReader reader,
        const CacheManagerContext& cacheManContext) {
        auto blobFileName = getBlobFile(blobHash, cacheManContext);
        if (FileUtils::fileExist(blobFileName)) {
            reader(std::ifstream(blobFileName, std::ios_base::binary));
        }

    }
};

FileStorageCacheManager::FileStorageCacheManager():
    m_impl(new FileStorageCacheManager::Impl()) {}

FileStorageCacheManager::~FileStorageCacheManager() = default;


void FileStorageCacheManager::writeCacheEntry(const std::string& id, ICacheManager::StreamWriter writer,
                                              const CacheManagerContext& cacheManContext) {
    m_impl->writeCacheEntry(id, writer, cacheManContext);
}

void FileStorageCacheManager::readCacheEntry(const std::string& id, FileStorageCacheManager::StreamReader reader,
                                             const CacheManagerContext& cacheManContext) {
    m_impl->readCacheEntry(id, reader, cacheManContext);
}

void FileStorageCacheManager::removeCacheEntry(const std::string& id, const CacheManagerContext& cacheManContext) {
    m_impl->removeCacheEntry(id, cacheManContext);
}

//===============CacheManagerContext=============


CacheManagerContext::CacheManagerContext(std::unique_ptr<CacheManagerContextImpl>&& internalImpl) : m_impl(std::move(internalImpl)) {}

CacheManagerContext::~CacheManagerContext() = default;

const Core& CacheManagerContext::getCore() const {
    return m_impl->getCore();
}

const std::string& CacheManagerContext::getDeviceName() const {
    return m_impl->getDeviceName();
}

const std::string& CacheManagerContext::getModelCacheDir() const {
    return m_impl->getModelCacheDir();
}

}  // namespace InferenceEngine
