# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import numpy as np

import spynnaker8 as sim
import spynnaker8.external_devices as ext

from utils.spikes_utils import populate_debug_times, read_recording_settings, read_spikes_input, neuron_id
from utils.debug_utils import cube_show_slider, receive_spikes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-O', '--live_output', action='store_true', default=False, help='Show visualisations')
    parser.add_argument('-V', '--vis', action='store_true', default=False, help='Show visualisations')

    parser.add_argument('-i', '--input', required=True, type=str, help='Text file with the spikes')

    args = parser.parse_args()

    return args


# def horizontal_connectivity(x, y, ):

# pass



# def vertical_connectivity():





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

    #### Set the first layers of the network

    sim_time = 2000

    # SpikeSourceArray for the positive polarity of the DVS
    stimulus_pos = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_pos), label='stimulus_pos')
    
    # SpikeSourceArray for the negative polarity of the DVS
    stimulus_neg = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_neg), label='stimulus_neg')

    # test_pop = sim.Population(n_total, sim.IF_curr_exp(), label='test_pop')

    # input_proj_pos = sim.Projection(stimulus_pos, test_pop, sim.OneToOneConnector(),\
    #                                 receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    # input_proj_neg = sim.Projection(stimulus_neg, test_pop, sim.OneToOneConnector(),\
    #                                 receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=5, delay=1))


    test_neuron = sim.Population(1, sim.IF_curr_exp(), label='test_neuron')

    pos_connections = [
        (neuron_id(15,16, cam_res), 0),
        (neuron_id(16,16, cam_res), 0),
        (neuron_id(17,16, cam_res), 0)
    ]

    projection_pos = sim.Projection(stimulus_pos, test_neuron, sim.FromListConnector(pos_connections), \
                                    receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    neg_connections = [
        (neuron_id(14,15, cam_res), 0),
        (neuron_id(15,15, cam_res), 0),
        (neuron_id(16,15, cam_res), 0),
        (neuron_id(17,15, cam_res), 0),
        (neuron_id(18,15, cam_res), 0),
        (neuron_id(14,17, cam_res), 0),
        (neuron_id(15,17, cam_res), 0),
        (neuron_id(16,17, cam_res), 0),
        (neuron_id(17,17, cam_res), 0),
        (neuron_id(18,17, cam_res), 0)
    ]

    projection_neg = sim.Projection(stimulus_pos, test_neuron, sim.FromListConnector(neg_connections), \
                                    receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=5, delay=1))

    # stimulus_pos.record(['spikes'])
    # stimulus_neg.record(['spikes'])
    test_neuron.record(['spikes', 'v'])




    sim.run(sim_time)


    # pos_spikes = stimulus_pos.get_data(variables=['spikes']).segments[0].spiketrains
    # neg_spikes = stimulus_neg.get_data(variables=['spikes']).segments[0].spiketrains
    # neo = test_pop.get_data(variables=['spikes'])
    # test_spikes = neo.segments[0].spiketrains
    # test_spikes = test_pop.get_data(variables=['spikes']).segments[0].spiketrains

    neo = test_neuron.get_data(variables=["spikes", "v"])
    test_spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]


    sim.end()



    # print(test_spikes)

    import pyNN.utility.plotting as plot
    import matplotlib.pyplot as matplotlib

    line_properties = [{'color': 'red', 'markersize': 2}, {'color': 'blue', 'markersize': 2}]
    plot.Figure(
        plot.Panel(v, ylabel="Membrane potential (mV)", data_labels=[test_neuron.label], yticks=True, xlim=(0, sim_time)),
        # plot.Panel(pos_spikes, ylabel='Neuron idx', yticks=True, xticks=True, markersize=5, xlim=(0, sim_time)),#, \
        # xlim=(0, sim_time), line_properties=line_properties), 
        # plot spikes (or in this case spike)
        plot.Panel(test_spikes, ylabel='Neuron idx', yticks=True, xticks=True, markersize=2, xlim=(0, sim_time)), 
        title="Spikes",
        annotations="Simulated with {}".format(sim.name())
    ) 
    matplotlib.show()
    

if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
