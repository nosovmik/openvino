# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import logging as log
from mo.front_ng.extractor import fe_user_data_repack


def moc_pipeline(argv: argparse.Namespace):
    from ngraph import Dimension, PartialShape    # pylint: disable=no-name-in-module,import-error
    log.info('New MOC pipeline')
    fem = argv.feManager
    log.info('fem.availableFrontEnds: ' + str(fem.availableFrontEnds()))
    log.info('Initializing new FE for framework {}'.format(argv.framework))
    fe = fem.loadByFramework(argv.framework)
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
    if len(user_shapes) > 0:
        inputsEqual = compare_nodes(inputModel.getInputs(), user_shapes)

    outputsEqual = True
    if len(outputs) > 0:
        outputsEqual = compare_nodes(inputModel.getOutputs(), outputs)
    log.debug("Inputs are same: {}, outputs are same: {}".format(inputsEqual, outputsEqual))

    if not inputsEqual and not outputsEqual:
        # Use ExtractSubgraph
        newInputPlaces = [x['node'] for x in user_shapes]
        newOutputPlaces = [x['node'] for x in outputs]
        log.debug("Using extract subgraph")
        log.debug("Inputs: {}".format(newInputPlaces))
        log.debug("Outputs: {}".format(newOutputPlaces))
        inputModel.extractSubgraph(newInputPlaces, newOutputPlaces)
    elif not inputsEqual:
        newInputPlaces = [x['node'] for x in user_shapes]
        log.debug("Using overrideAllInputs")
        log.debug("Inputs: {}".format(newInputPlaces))
        inputModel.overrideAllInputs(newInputPlaces)
    elif not outputsEqual:
        newOutputPlaces = [x['node'] for x in outputs]
        log.debug("Using overrideAllOutputs")
        log.debug("Outputs: {}".format(newOutputPlaces))
        inputModel.overrideAllOutputs(newOutputPlaces)

    # TODO: handle element type
    if len(user_shapes) > 0:
        for user_shape in user_shapes:
            if 'shape' in user_shape and user_shape['shape'] is not None:
                inputModel.setPartialShape(user_shape['node'], PartialShape(user_shape['shape']))
    nGraphFunction = fe.convert(inputModel)

    # Set batch size
    # TODO: consider alternatives like exposing nGraph transformatons or use IENetwork.reshape to set batch size
    if argv.batch > 0:
        log.debug("Setting batch size to {}".format(argv.batch))
        for par in nGraphFunction.get_parameters():
            oldPartShape = par.get_partial_shape()
            newshape = []
            if oldPartShape.rank.is_static:
                for i in range(oldPartShape.rank.get_length()):
                    # TODO: Assume batch size is always 1-st dimension in shape
                    # Keep other dimentions unchanged
                    newshape.append(Dimension(argv.batch) if i is 0 else oldPartShape.get_dimension(i))
            else:
                raise Exception("Setting batch size for shapes with dynamic rank is not supported")
            newPartShape = PartialShape(newshape)
            log.debug("Input: {}, Old shape: {}, New shape: {}".format(par.get_friendly_name(), oldPartShape, newPartShape))
            par.set_partial_shape(newPartShape)
    nGraphFunction.validate_nodes_and_infer_types()
    return nGraphFunction