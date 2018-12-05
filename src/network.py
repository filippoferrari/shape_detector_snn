# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import numpy as np

import spynnaker8 as sim
import spynnaker8.external_devices as ext

from utils.spikes_utils import populate_debug_times, read_recording_settings, read_spikes_input
from utils.debug_utils import cube_show_slider, receive_spikes


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
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)

    # Set the first layer of the network
    spike_array_pos = {'spike_times': spikes_pos}
    stimulus_pos = sim.Population(n_total, sim.SpikeSourceArray, spike_array_pos, label='stimulus_pos')

    spike_array_neg = {'spike_times': spikes_neg}
    stimulus_neg = sim.Population(n_total, sim.SpikeSourceArray, spike_array_neg, label='stimulus_neg')

    test_pop = sim.Population(n_total, sim.IF_curr_exp(), label="test_pop")
    input_proj_pos = sim.Projection(stimulus_pos, test_pop, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=5, delay=1))
    input_proj_neg = sim.Projection(stimulus_neg, test_pop, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=-1, delay=1))


    if args.live_output:
        # Activate live output from the first layer - testing only
        ext.activate_live_output_for(stimulus_pos, database_notify_port_num=19996)
        live_spikes_connection_pos = sim.external_devices.SpynnakerLiveSpikesConnection(receive_labels=["stimulus_pos"], local_port=19996)
        live_spikes_connection_pos.add_receive_callback("stimulus_pos", receive_spikes)

        ext.activate_live_output_for(stimulus_neg, database_notify_port_num=19997)
        live_spikes_connection_neg = sim.external_devices.SpynnakerLiveSpikesConnection(receive_labels=["stimulus_neg"], local_port=19997)
        live_spikes_connection_neg.add_receive_callback("stimulus_neg", receive_spikes)

    # stimulus_pos.record(['spikes'])
    # stimulus_neg.record(['spikes'])
    test_pop.record(['spikes'])

    # pos_spikes = stimulus_pos.get_data(variables=['spikes']).segments[0].spiketrains
    # neg_spikes = stimulus_neg.get_data(variables=['spikes']).segments[0].spiketrains
    test_spikes = test_pop.get_data(variables=['spikes']).segments[0].spiketrains

    sim.run(sim_time)

    sim.end()


    import pyNN.utility.plotting as plot
    import matplotlib.pyplot as matplotlib

    line_properties = [{'color': 'red', 'markersize': 2}, {'color': 'blue', 'markersize': 2}]
    plot.Figure(
        # plot voltage for first ([0]) neuron
        # plot.Panel(pos_spikes, neg_spikes, ylabel='Neuron idx', yticks=True, xticks=True, \
        #            xlim=(0, sim_time), line_properties=line_properties), 
        # plot spikes (or in this case spike)
        plot.Panel(test_spikes, ylabel='Neuron idx', yticks=True, xticks=True, markersize=5, xlim=(0, sim_time)), 
        title="Spikes",
        annotations="Simulated with {}".format(sim.name())
    ) 
    matplotlib.show()
    

if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
