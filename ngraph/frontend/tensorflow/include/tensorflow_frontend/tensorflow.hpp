// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: include it by just frontend_manager.hpp without path
//#include "../../include/frontend_manager/frontend_manager.hpp"
#include "frontend_manager/frontend_manager.hpp"

namespace tensorflow { class GraphDef; }

namespace ngraph
{
    namespace frontend
    {
        class PlaceTensorflow : public Place
        {
        public:

            std::string name;
            enum Kind { PORT_INPUT, PORT_OUTPUT, TENSOR, OP } kind;
            size_t port;

            PlaceTensorflow (const std::string& _name, Kind _kind = OP, size_t _port = 0) : name(_name), kind(_kind), port(_port) {}
        };

        class NGRAPH_API InputModelTensorflow : public InputModel
        {
        public:

            std::shared_ptr<tensorflow::GraphDef> graph_def;
            std::string path;

            // TODO: map from PlaceTensorflow, not from name string
            std::map<std::string, ngraph::PartialShape> partialShapes;

            InputModelTensorflow (const std::string& _path);

            std::vector<Place::Ptr> getInputs () const override;

            void setPartialShape (Place::Ptr place, const ngraph::PartialShape& pshape) override;
        };

        class NGRAPH_API FrontEndTensorflow : public FrontEnd
        {
        public:

            FrontEndTensorflow ()
            {
            }

            virtual InputModel::Ptr loadFromFile (const std::string& path) const override
            {
                return std::make_shared<InputModelTensorflow>(path);
            }

            virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const override;
        };

    } // namespace frontend

} // namespace ngraph
