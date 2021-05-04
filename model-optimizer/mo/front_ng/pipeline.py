# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import logging as log
from mo.front_ng.extractor import fe_user_data_repack


def moc_pipeline(argv: argparse.Namespace):
    from ngraph import function_to_cnn                # pylint: disable=no-name-in-module,import-error
    from ngraph import PartialShape                   # pylint: disable=no-name-in-module,import-error
    from ngraph.utils.types import get_element_type   # pylint: disable=no-name-in-module,import-error
    log.info('New MOC pipeline')
    fem = argv.feManager
    log.info(f'fem.availableFrontEnds: {str(fem.availableFrontEnds())}')
    log.info(f'Initializing new FE for framework {argv.framework}')
    fe = fem.loadByFramework(argv.framework)
    print(fe)
    inputModel = fe.loadFromFile(argv.input_model)

    user_shapes, outputs, freeze_placeholder = fe_user_data_repack(
        inputModel, argv.placeholder_shapes, argv.placeholder_data_types,
        argv.output, argv.freeze_placeholder_with_value)

    def compare_nodes(old, new):
        eq = len(old) == len(new)
        if eq:
            for item in old:
                found = [x for x in new if x['node'].isEqual(item)]
                if not found:
                    eq = False
                    break
        return eq

    inputsEqual = True
    if user_shapes:
        inputsEqual = compare_nodes(inputModel.getInputs(), user_shapes)

    outputsEqual = True
    if outputs:
        outputsEqual = compare_nodes(inputModel.getOutputs(), outputs)
    print("Inputs are same: {}, outputs are same: {}".format(inputsEqual, outputsEqual))

    if not inputsEqual and not outputsEqual:
        # Use ExtractSubgraph
        newInputPlaces = [x['node'] for x in user_shapes]
        newOutputPlaces = [x['node'] for x in outputs]
        print("Using extract subgraph")
        print("Inputs: {}".format(newInputPlaces))
        print("Outputs: {}".format(newOutputPlaces))
        inputModel.extractSubgraph(newInputPlaces, newOutputPlaces)
    elif not inputsEqual:
        newInputPlaces = [x['node'] for x in user_shapes]
        print("Using overrideAllInputs")
        print("Inputs: {}".format(newInputPlaces))
        inputModel.overrideAllInputs(newInputPlaces)
    elif not outputsEqual:
        newOutputPlaces = [x['node'] for x in outputs]
        print("Using overrideAllOutputs")
        print("Outputs: {}".format(newOutputPlaces))
        inputModel.overrideAllOutputs(newOutputPlaces)

    if user_shapes:
        for user_shape in user_shapes:
            if 'shape' in user_shape and user_shape['shape'] is not None:
                inputModel.setPartialShape(user_shape['node'], PartialShape(user_shape['shape']))
            if 'data_type' in user_shape and user_shape['data_type'] is not None:
                data_type = get_element_type(user_shape['data_type'])
                log.debug(f"Set data type: {data_type}")
                inputModel.setElementType(user_shape['node'], data_type)

    nGraphModel = fe.convert(inputModel)
    network = function_to_cnn(nGraphModel)
    return network
