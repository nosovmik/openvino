# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import logging as log
from mo.front_ng.extractor import fe_user_data_repack
from mo.middle.passes.infer import validate_batch_in_shape

from ngraph import Dimension, PartialShape        # pylint: disable=no-name-in-module,import-error
from ngraph.utils.types import get_element_type   # pylint: disable=no-name-in-module,import-error

def moc_pipeline(argv: argparse.Namespace):
    fem = argv.feManager
    log.info('Available front ends: {}'.format(str(fem.get_available_front_ends())))
    log.info('Initializing new FE for framework {}'.format(argv.framework))
    fe = fem.load_by_framework(argv.framework)
    input_model = fe.load_from_file(argv.input_model)

    user_shapes, outputs, freeze_placeholder = fe_user_data_repack(
        input_model, argv.placeholder_shapes, argv.placeholder_data_types,
        argv.output, argv.freeze_placeholder_with_value)

    def compare_nodes(old, new):
        eq = len(old) == len(new)
        if eq:
            for item in old:
                found = [x for x in new if x['node'].is_equal(item)]
                if not found:
                    eq = False
                    break
        return eq

    inputs_equal = True
    if user_shapes:
        inputs_equal = compare_nodes(input_model.get_inputs(), user_shapes)

    outputs_equal = True
    if outputs:
        outputs_equal = compare_nodes(input_model.get_outputs(), outputs)
    log.debug('Inputs are same: {}, outputs are same: {}'.format(inputs_equal, outputs_equal))

    if not inputs_equal and not outputs_equal:
        # Use ExtractSubgraph
        new_input_places = [x['node'] for x in user_shapes]
        new_output_places = [x['node'] for x in outputs]
        log.debug('Using extract subgraph')
        input_model.extract_subgraph(new_input_places, new_output_places)
    elif not inputs_equal:
        new_input_places = [x['node'] for x in user_shapes]
        log.debug('Using override_all_inputs')
        input_model.override_all_inputs(new_input_places)
    elif not outputs_equal:
        new_output_places = [x['node'] for x in outputs]
        log.debug('Using override_all_outputs')
        input_model.override_all_outputs(new_output_places)

    if user_shapes:
        for user_shape in user_shapes:
            if 'shape' in user_shape and user_shape['shape'] is not None:
                input_model.set_partial_shape(user_shape['node'], PartialShape(user_shape['shape']))
            if 'data_type' in user_shape and user_shape['data_type'] is not None:
                data_type = get_element_type(user_shape['data_type'])
                log.debug('Set data type: {}'.format(data_type))
                input_model.set_element_type(user_shape['node'], data_type)

    # Set batch size
    if argv.batch is not None and argv.batch > 0:
        log.debug('Setting batch size to {}'.format(argv.batch))
        for place in input_model.get_inputs():
            old_partial_shape = input_model.get_partial_shape(place)
            new_shape = []
            old_shape_converted = []
            joined_name = ' '.join(place.get_names())
            if old_partial_shape.rank.is_static:
                for i in range(old_partial_shape.rank.get_length()):
                    # Assume batch size is always 1-st dimension in shape
                    # Keep other dimensions unchanged
                    new_shape.append(Dimension(argv.batch) if i == 0 else old_partial_shape.get_dimension(i))
                    old_shape_converted.append(old_partial_shape.get_dimension(i))

                validate_batch_in_shape(old_shape_converted, joined_name)
            else:
                # In case of fully dynamic shape raise the same error as for invalid batch dimension
                validate_batch_in_shape(old_shape_converted, joined_name)

            new_partial_shape = PartialShape(new_shape)
            log.debug('Input: {}, Old shape: {}, New shape: {}'.format(joined_name, old_shape_converted, new_shape))
            input_model.set_partial_shape(place, new_partial_shape)

    ngraph_function = fe.convert(input_model)
    return ngraph_function
