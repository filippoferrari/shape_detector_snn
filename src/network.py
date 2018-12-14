# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import matplotlib.pyplot as matplotlib
import numpy as np

import pyNN.utility.plotting as plot

import spynnaker8 as sim
import spynnaker8.external_devices as ext

from utils.debug_utils import cube_show_slider, receive_spikes
from utils.network_utils import horizontal_connectivity_pos, horizontal_connectivity_neg, \
                                vertical_connectivity_pos, vertical_connectivity_neg, \
                                left_diagonal_connectivity_pos, left_diagonal_connectivity_neg, \
                                right_diagonal_connectivity_pos, right_diagonal_connectivity_neg
from utils.spikes_utils import populate_debug_times, read_recording_settings, read_spikes_input, neuron_id


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-O', '--live_output', action='store_true', default=False, help='Show visualisations')
    parser.add_argument('-V', '--vis', action='store_true', default=False, help='Show visualisations')

    parser.add_argument('-i', '--input', required=True, type=str, help='Text file with the spikes')

    args = parser.parse_args()

    return args


def main(args):
    # Read the input file
    raw_spikes, cam_res, sim_time = read_recording_settings(args)

    # Spikes decoded
    spikes_pos, spikes_neg = read_spikes_input(raw_spikes, cam_res, sim_time)

    if args.vis:
        times_debug = populate_debug_times(raw_spikes, cam_res, sim_time)
        cube_show_slider(times_debug)

    n_total = cam_res * cam_res

    sim.setup(timestep=1.0)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 150)


    ##########################################################
    #### Set the first layers of the network

    # SpikeSourceArray for the positive polarity of the DVS
    stimulus_pos = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_pos), label='stimulus_pos')
    
    # SpikeSourceArray for the negative polarity of the DVS
    stimulus_neg = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_neg), label='stimulus_neg')


    ##########################################################
    #### Horizontal receptive field
    horizontal_layer = sim.Population(n_total / 4, sim.IF_curr_exp(), label='horizontal_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, 2):
        for y in range(0, cam_res, 2):
            pos_connections += horizontal_connectivity_pos(cam_res, x, y, cam_res/2)
            neg_connections += horizontal_connectivity_neg(cam_res, x, y, cam_res/2)

    horizontal_proj_pos = sim.Projection(stimulus_pos, horizontal_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    horizontal_proj_neg = sim.Projection(stimulus_neg, horizontal_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    horizontal_layer.record(['spikes'])


    ##########################################################
    #### Vertical receptive field
    vertical_layer = sim.Population(n_total / 4, sim.IF_curr_exp(), label='horizontal_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, 2):
        for y in range(0, cam_res, 2):
            pos_connections += vertical_connectivity_pos(cam_res, x, y, cam_res/2)
            neg_connections += vertical_connectivity_neg(cam_res, x, y, cam_res/2)

    vertical_proj_pos = sim.Projection(stimulus_pos, vertical_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    vertical_proj_neg = sim.Projection(stimulus_neg, vertical_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    vertical_layer.record(['spikes'])


    ##########################################################
    #### Left diagonal receptive field
    left_diag_layer = sim.Population(n_total / 4, sim.IF_curr_exp(), label='horizontal_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, 2):
        for y in range(0, cam_res, 2):
            pos_connections += left_diagonal_connectivity_pos(cam_res, x, y, cam_res/2)
            neg_connections += left_diagonal_connectivity_neg(cam_res, x, y, cam_res/2)

    left_diag_proj_pos = sim.Projection(stimulus_pos, left_diag_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    left_diag_proj_neg = sim.Projection(stimulus_neg, left_diag_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    left_diag_layer.record(['spikes'])


    ##########################################################
    #### Right diagonal receptive field
    right_diag_layer = sim.Population(n_total / 4, sim.IF_curr_exp(), label='horizontal_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, 2):
        for y in range(0, cam_res, 2):
            pos_connections += right_diagonal_connectivity_pos(cam_res, x, y, cam_res/2)
            neg_connections += right_diagonal_connectivity_neg(cam_res, x, y, cam_res/2)

    right_diag_proj_pos = sim.Projection(stimulus_pos, right_diag_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    right_diag_proj_neg = sim.Projection(stimulus_neg, right_diag_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    right_diag_layer.record(['spikes'])


    ##########################################################
    #### Run the simulation
    sim.run(sim_time)

    neo = horizontal_layer.get_data(variables=['spikes'])
    horizontal_spikes = neo.segments[0].spiketrains

    neo = vertical_layer.get_data(variables=['spikes'])
    vertical_spikes = neo.segments[0].spiketrains
    
    neo = left_diag_layer.get_data(variables=['spikes'])
    left_diag_spikes = neo.segments[0].spiketrains
    
    neo = right_diag_layer.get_data(variables=['spikes'])
    right_diag_spikes = neo.segments[0].spiketrains

    sim.end()


    ##########################################################
    #### Plot the receptive fields
    line_properties = [{'color': 'red', 'markersize': 2}, {'color': 'blue', 'markersize': 2}]
    plot.Figure(
        # plot.Panel(v, ylabel="Membrane potential (mV)", data_labels=[test_neuron.label], yticks=True, xlim=(0, sim_time)),
        # plot.Panel(pos_spikes, ylabel='Neuron idx', yticks=True, xticks=True, markersize=5, xlim=(0, sim_time)),#, \
        # xlim=(0, sim_time), line_properties=line_properties), 
        # plot spikes (or in this case spike)
        plot.Panel(horizontal_spikes, ylabel='Neuron idx', yticks=True, xlabel='Horizontal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(vertical_spikes, ylabel='Neuron idx', yticks=True, xlabel='Vertical', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(left_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Left diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(right_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Right diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        title='Receptive fields',
        annotations='Simulated with {}'.format(sim.name())
    ) 
    matplotlib.show()
    

if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
