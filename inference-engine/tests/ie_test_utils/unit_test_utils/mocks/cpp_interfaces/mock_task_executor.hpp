// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <gmock/gmock.h>
#include <threading/ie_itask_executor.hpp>

class MockTaskExecutor : public InferenceEngine::ITaskExecutor {
public:
    typedef std::shared_ptr<MockTaskExecutor> Ptr;

    MOCK_METHOD(void, run, (InferenceEngine::Task));
};
