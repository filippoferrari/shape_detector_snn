# -*- coding: utf-8 -*-

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import pyNN.utility.plotting as plot

from src.dvs_emulator import DVS_Emulator

import spynnaker8 as sim
import spynnaker8.external_devices as ext

from src.utils.constants import OUTPUT_RATE, OUTPUT_TIME, OUTPUT_TIME_BIN_THR, KEY_SPINNAKER, KEY_XYP

from src.utils.debug_utils import receive_spikes, image_slice_viewer

from src.utils.io_utils import parse_args, read_config, save_video

from src.utils.spikes_utils import read_spikes_from_video, populate_debug_times_from_video, coord_from_neuron, \
                                   read_recording_settings, neuron_id, populate_debug_times

from src.network_utils.receptive_fields import horizontal_connectivity_pos, horizontal_connectivity_neg, \
                                           vertical_connectivity_pos, vertical_connectivity_neg, \
                                           left_diagonal_connectivity_pos, left_diagonal_connectivity_neg, \
                                           right_diagonal_connectivity_pos, right_diagonal_connectivity_neg

from src.network_utils.shapes import hor_connections, vert_connections, left_diag_connections, right_diag_connections


def main(config):
    # For some weird opencv/matplotlib bug, need to call matplotlib before opencv
    plt.plot([1,2,3])
    plt.close('all')

    if config['video']:
        spikes_pos, spikes_neg, cam_res, sim_time = read_spikes_from_video(config['input'])
    else:
        cam_res = 32
        dvs = DVS_Emulator(cam_res, config)

        dvs.read_video_source()

        spikes_pos, spikes_neg = dvs.split_pos_neg_spikes()
        cam_res = dvs.cam_res
        sim_time = dvs.sim_time

        if config['output_file']:
            dvs.save_output(config['output_file'])

    #### Display input spikes
    if config['vis']:
        if config['video']:
            vis_spikes = populate_debug_times_from_video(spikes_pos, spikes_neg, cam_res, sim_time)
            image_slice_viewer(vis_spikes)
        else:
            image_slice_viewer(dvs.tuple_to_numpy(), step=dvs.time_bin_ms)


    n_neurons = cam_res * cam_res

    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 120)

    ##########################################################
    #### Some values for the network

    # Some values for the network 
    exc_weight = config['exc_weight']
    exc_delay = config['exc_delay']

    inh_weight = config['inh_weight']
    inh_delay = config['inh_delay']

    shapes_weight = config['shapes_weight']
    shapes_delay = config['shapes_delay']

    down_size = config['down_size']

    ##########################################################
    #### Set the first layers of the network

    # SpikeSourceArray for the positive polarity of the DVS
    stimulus_pos = sim.Population(n_neurons, sim.SpikeSourceArray(spike_times=spikes_pos), label='stimulus_pos')
    stimulus_pos.record(['spikes'])

    # SpikeSourceArray for the negative polarity of the DVS
    stimulus_neg = sim.Population(n_neurons, sim.SpikeSourceArray(spike_times=spikes_neg), label='stimulus_neg')


    ####################################################################################################################
    #### RECEPTIVE FIELDS
    ####################################################################################################################


    ##########################################################
    #### Horizontal receptive field
    horizontal_layer = sim.Population(n_neurons / (down_size * down_size), sim.IF_curr_exp(), label='horizontal_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += horizontal_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += horizontal_connectivity_neg(cam_res, x, y, cam_res/down_size)

    sim.Projection(stimulus_pos, horizontal_layer, sim.FromListConnector(pos_connections), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    sim.Projection(stimulus_neg, horizontal_layer, sim.FromListConnector(neg_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    horizontal_layer.record(['spikes'])


    ##########################################################
    #### Vertical receptive field
    vertical_layer = sim.Population(n_neurons / (down_size * down_size), sim.IF_curr_exp(), label='vertical_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += vertical_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += vertical_connectivity_neg(cam_res, x, y, cam_res/down_size)

    sim.Projection(stimulus_pos, vertical_layer, sim.FromListConnector(pos_connections), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    sim.Projection(stimulus_neg, vertical_layer, sim.FromListConnector(neg_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    vertical_layer.record(['spikes'])


    ##########################################################
    #### Left diagonal receptive field
    left_diag_layer = sim.Population(n_neurons / (down_size * down_size), sim.IF_curr_exp(), label='left_diag_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += left_diagonal_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += left_diagonal_connectivity_neg(cam_res, x, y, cam_res/down_size)

    sim.Projection(stimulus_pos, left_diag_layer, sim.FromListConnector(pos_connections), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    sim.Projection(stimulus_neg, left_diag_layer, sim.FromListConnector(neg_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    left_diag_layer.record(['spikes'])


    ##########################################################
    #### Right diagonal receptive field
    right_diag_layer = sim.Population(n_neurons / (down_size * down_size), sim.IF_curr_exp(), label='right_diag_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += right_diagonal_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += right_diagonal_connectivity_neg(cam_res, x, y, cam_res/down_size)

    sim.Projection(stimulus_pos, right_diag_layer, sim.FromListConnector(pos_connections), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    sim.Projection(stimulus_neg, right_diag_layer, sim.FromListConnector(neg_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    right_diag_layer.record(['spikes'])



    ####################################################################################################################
    #### SHAPES DETECTORS
    ####################################################################################################################

    inhibition_exc_w = 3
    inhibition_delay = 1

    ##########################################################
    #### Square shape detector
    square_layer = sim.Population(n_neurons / (down_size * down_size), sim.IF_curr_exp(), label='square_layer')
    # The sides of the square are of length 2 * stride + 1
    stride = 2

    connections = [] 
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            connections += hor_connections(cam_res/down_size, x, y, stride, cam_res/down_size, shapes_weight, shapes_delay)

    sim.Projection(horizontal_layer, square_layer, sim.FromListConnector(connections, column_names=['pre', 'post', 'weight', 'delay']), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=shapes_weight, delay=shapes_delay))

    connections = [] 
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            connections += vert_connections(cam_res/down_size, x, y, stride, cam_res/down_size, shapes_weight, shapes_delay)

    sim.Projection(vertical_layer, square_layer, sim.FromListConnector(connections, column_names=['pre', 'post', 'weight', 'delay']), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=shapes_weight, delay=shapes_delay))


    # Lateral inhibition
    lateral_inh_connections = []
    for i in range(0, n_neurons / (down_size * down_size)):
        for j in range(0, n_neurons / (down_size * down_size)):
            if i != j:
                lateral_inh_connections.append((i, j))

    sim.Projection(square_layer, square_layer, sim.FromListConnector(lateral_inh_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inhibition_exc_w, delay=inhibition_delay))


    ##########################################################
    #### Diamond shape detector
    diamond_layer = sim.Population(n_neurons / (down_size * down_size), sim.IF_curr_exp(), label='diamond_layer')
    # The sides of the diamond are of length 2 * stride + 1
    stride = 2

    connections = [] 
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            connections += left_diag_connections(cam_res/down_size, x, y, stride, cam_res/down_size, shapes_weight, shapes_delay)

    sim.Projection(left_diag_layer, diamond_layer, sim.FromListConnector(connections, column_names=['pre', 'post', 'weight', 'delay']), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=shapes_weight, delay=shapes_delay))

    connections = [] 
    for x in range(0, cam_res/down_size):
        for y in range(0, cam_res/down_size):
            connections += right_diag_connections(cam_res/down_size, x, y, stride, cam_res/down_size, shapes_weight, shapes_delay)

    sim.Projection(right_diag_layer, diamond_layer, sim.FromListConnector(connections, column_names=['pre', 'post', 'weight', 'delay']), \
                   receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=shapes_weight, delay=shapes_delay))


    # Lateral inhibition
    lateral_inh_connections = []
    for i in range(0, n_neurons / (down_size * down_size)):
        for j in range(0, n_neurons / (down_size * down_size)):
            if i != j:
                lateral_inh_connections.append((i, j))

    sim.Projection(diamond_layer, diamond_layer, sim.FromListConnector(lateral_inh_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inhibition_exc_w, delay=inhibition_delay))


    # ##########################################################
    # #### Inhibition between shapes
    # shapes_inhibition = []
    # for i in range(0, n_neurons / (down_size * down_size)):
    #     shapes_inhibition.append((i, i))
    # sim.Projection(square_layer, diamond_layer, sim.FromListConnector(shapes_inhibition), \
    #                receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=10, delay=1))
    # sim.Projection(diamond_layer, square_layer, sim.FromListConnector(shapes_inhibition), \
    #                receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=3, delay=1))

    square_layer.record(['spikes'])
    diamond_layer.record(['spikes'])

    ##########################################################
    #### Run the simulation
    sim.run(sim_time)

    # neo = stimulus_pos.get_data(variables=['spikes'])
    # stimulus_pos_spikes = neo.segments[0].spiketrains

    neo = horizontal_layer.get_data(variables=['spikes'])
    horizontal_spikes = neo.segments[0].spiketrains

    neo = vertical_layer.get_data(variables=['spikes'])
    vertical_spikes = neo.segments[0].spiketrains
    
    neo = left_diag_layer.get_data(variables=['spikes'])
    left_diag_spikes = neo.segments[0].spiketrains
    
    neo = right_diag_layer.get_data(variables=['spikes'])
    right_diag_spikes = neo.segments[0].spiketrains

    neo = square_layer.get_data(variables=['spikes'])
    square_spikes = neo.segments[0].spiketrains

    neo = diamond_layer.get_data(variables=['spikes'])
    diamond_spikes = neo.segments[0].spiketrains

    sim.end()


    ##########################################################
    #### Plot the receptive fields
    # line_properties = [{'color': 'red', 'markersize': 2}, {'color': 'blue', 'markersize': 2}]
    plot.Figure(
        # plot.Panel(v, ylabel="Membrane potential (mV)", data_labels=[test_neuron.label], yticks=True, xlim=(0, sim_time)),
        # plot.Panel(pos_spikes, ylabel='Neuron idx', yticks=True, xticks=True, markersize=5, xlim=(0, sim_time)),#, \
        # xlim=(0, sim_time), line_properties=line_properties), 
        # plot spikes (or in this case spike)
        # plot.Panel(stimulus_pos_spikes, ylabel='Neuron idx', yticks=True, xlabel='Pos', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(horizontal_spikes, ylabel='Neuron idx', yticks=True, xlabel='Horizontal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(vertical_spikes, ylabel='Neuron idx', yticks=True, xlabel='Vertical', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(left_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Left diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(right_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Right diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        title='Receptive fields',
        annotations='Simulated with {}\n {}'.format(sim.name(), config['input'])
    ) 
    plt.show()
    
    plot.Figure(
        plot.Panel(square_spikes, ylabel='Neuron idx', yticks=True, xlabel='Square shape', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(diamond_spikes, ylabel='Neuron idx', yticks=True, xlabel='Diamond shape', xticks=True, markersize=2, xlim=(0, sim_time)), 
        title='Shape detector',
        annotations='Simulated with {}\n {}'.format(sim.name(), config['input'])
    )
    plt.show()


    if not config['dont_save']:
        # Process spiketrains for each shape
        spiking_times_square = shape_spikes_bin(square_spikes)
        spiking_times_diamond = shape_spikes_bin(diamond_spikes)

        if config['webcam']:
            save_video(config, dvs.video_writer_path, [spiking_times_square,spiking_times_diamond], stride, ['r','y'])
        else:
            save_video(config, config['input'], [spiking_times_square,spiking_times_diamond], stride, ['r','y'])


def shape_spikes_bin(shape_spikes):
    spiking_times = {}
    for neuron, spikes in enumerate(shape_spikes):
        for spike in spikes:
            if not int(spike) in spiking_times:
                spiking_times[int(spike)] = []
            spiking_times[int(spike)].append(neuron)
    return spiking_times


if __name__ == '__main__':
    args_parsed = parse_args()
    config = read_config(args_parsed)
    main(config)
