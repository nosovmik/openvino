// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include "place.hpp"

namespace ngraph {
namespace frontend {

class NGRAPH_API InputModelPDPD : public InputModel
{
    friend class FrontEndPDPD;
    class InputModelPDPDImpl;
    std::shared_ptr<InputModelPDPDImpl> _impl;

    std::vector<float> readWeight(const std::string& name, int64_t tensor_length);
    std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces(int i) const;
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarPlaces(int i) const;
    size_t getBlockNumber() const;

public:
    explicit InputModelPDPD (const std::string& _path);
    std::vector<Place::Ptr> getInputs () const override;
    std::vector<Place::Ptr> getOutputs () const override;
    Place::Ptr getPlaceByTensorName (const std::string& tensorName) const override;
    void overrideAllOutputs (const std::vector<Place::Ptr>& outputs) override;
    void overrideAllInputs (const std::vector<Place::Ptr>& inputs) override;
    void extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;
    void setDefaultShape (Place::Ptr place, const ngraph::Shape&) override;
    void setPartialShape (Place::Ptr place, const ngraph::PartialShape&) override;
    void setElementType (Place::Ptr place, const ngraph::element::Type&) override;

};

} // namespace frontend
} // namespace ngraph
