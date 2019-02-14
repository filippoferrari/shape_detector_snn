# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import itertools
import matplotlib.pyplot as matplotlib
import numpy as np

import cv2

import pyNN.utility.plotting as plot

import spynnaker8 as sim
import spynnaker8.external_devices as ext

from utils.debug_utils import receive_spikes, image_slice_viewer
from utils.spikes_utils import read_spikes_from_video, populate_debug_times_from_video, coord_from_neuron#, read_recording_settings, read_spikes_input, neuron_id

from network_utils.receptive_fields import horizontal_connectivity_pos, horizontal_connectivity_neg, \
                                           vertical_connectivity_pos, vertical_connectivity_neg, \
                                           left_diagonal_connectivity_pos, left_diagonal_connectivity_neg, \
                                           right_diagonal_connectivity_pos, right_diagonal_connectivity_neg
from network_utils.shapes import hor_connections, vert_connections, left_diag_connections, right_diag_connections

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-O', '--live_output', action='store_true', default=False, help='Show visualisations')
    parser.add_argument('-V', '--vis', action='store_true', default=False, help='Show visualisations')

    parser.add_argument('-i', '--input', required=True, type=str, help='Text file with the spikes')

    args = parser.parse_args()

    return args


def main(args):
    # # Read the input file
    # raw_spikes, cam_res, sim_time = read_recording_settings(args)

    # # Spikes decoded
    # spikes_pos, spikes_neg = read_spikes_input(raw_spikes, cam_res, sim_time)

    spikes_pos, spikes_neg, cam_res, sim_time = read_spikes_from_video(args.input)

    if args.vis:
        # times_debug = populate_debug_times(raw_spikes, cam_res, sim_time)
        times_debug = populate_debug_times_from_video(spikes_pos, cam_res, sim_time)
        image_slice_viewer(times_debug)
        times_debug = populate_debug_times_from_video(spikes_neg, cam_res, sim_time)
        image_slice_viewer(times_debug)

    n_total = cam_res * cam_res

    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

    # Some values for the network 
    exc_weight = 3
    inh_weight = 1

    exc_delay = 1
    inh_delay = 1

    down_size = 1

    ##########################################################
    #### Set the first layers of the network

    # SpikeSourceArray for the positive polarity of the DVS
    stimulus_pos = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_pos), label='stimulus_pos')
    
    # SpikeSourceArray for the negative polarity of the DVS
    stimulus_neg = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_neg), label='stimulus_neg')


    ##########################################################
    #### Horizontal receptive field
    horizontal_layer = sim.Population(n_total / (down_size*down_size), sim.IF_curr_exp(), label='horizontal_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += horizontal_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += horizontal_connectivity_neg(cam_res, x, y, cam_res/down_size)

    horizontal_proj_pos = sim.Projection(stimulus_pos, horizontal_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    horizontal_proj_neg = sim.Projection(stimulus_neg, horizontal_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    horizontal_layer.record(['spikes'])


    ##########################################################
    #### Vertical receptive field
    vertical_layer = sim.Population(n_total / (down_size*down_size), sim.IF_curr_exp(), label='vertical_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += vertical_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += vertical_connectivity_neg(cam_res, x, y, cam_res/down_size)

    vertical_proj_pos = sim.Projection(stimulus_pos, vertical_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    vertical_proj_neg = sim.Projection(stimulus_neg, vertical_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    vertical_layer.record(['spikes'])


    ##########################################################
    #### Left diagonal receptive field
    left_diag_layer = sim.Population(n_total / (down_size*down_size), sim.IF_curr_exp(), label='left_diag_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += left_diagonal_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += left_diagonal_connectivity_neg(cam_res, x, y, cam_res/down_size)

    left_diag_proj_pos = sim.Projection(stimulus_pos, left_diag_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    left_diag_proj_neg = sim.Projection(stimulus_neg, left_diag_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    left_diag_layer.record(['spikes'])


    ##########################################################
    #### Right diagonal receptive field
    right_diag_layer = sim.Population(n_total / (down_size*down_size), sim.IF_curr_exp(), label='right_diag_layer')

    pos_connections = [] 
    neg_connections = []
    for x in range(0, cam_res, down_size):
        for y in range(0, cam_res, down_size):
            pos_connections += right_diagonal_connectivity_pos(cam_res, x, y, cam_res/down_size)
            neg_connections += right_diagonal_connectivity_neg(cam_res, x, y, cam_res/down_size)

    right_diag_proj_pos = sim.Projection(stimulus_pos, right_diag_layer, sim.FromListConnector(pos_connections), \
                                        receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    right_diag_proj_neg = sim.Projection(stimulus_neg, right_diag_layer, sim.FromListConnector(neg_connections), \
                                        receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay))

    right_diag_layer.record(['spikes'])


    ##########################################################
    #### Square shape detector
    square_layer = sim.Population(n_total / (down_size * down_size), sim.IF_curr_exp(), label='square_layer')
    # The sides of the square are of length 2 * stride + 1
    stride = 2

    pos_connections = [] 
    for x in range(0, cam_res/down_size):
        for y in range(0, cam_res/down_size):
            pos_connections += hor_connections(cam_res/down_size, x, y, stride, cam_res/down_size)

    square_hor = sim.Projection(horizontal_layer, square_layer, sim.FromListConnector(pos_connections), \
                                receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    pos_connections = [] 
    for x in range(0, cam_res/down_size):
        for y in range(0, cam_res/down_size):
            pos_connections += vert_connections(cam_res/down_size, x, y, stride, cam_res/down_size)

    square_vert = sim.Projection(vertical_layer, square_layer, sim.FromListConnector(pos_connections), \
                                    receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))


    # Lateral inhibition
    lateral_inh_connections = []
    for i in range(0, n_total / (down_size * down_size)):
        for j in range(0, n_total / (down_size * down_size)):
            if i != j:
                lateral_inh_connections.append((i, j))

    lat_inh = sim.Projection(square_layer, square_layer, sim.FromListConnector(lateral_inh_connections), \
                             receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=3, delay=1))


    square_layer.record(['spikes'])


    # ##########################################################
    # #### Diamond shape detector
    # diamond_layer = sim.Population(n_total/4, sim.IF_curr_exp(), label='diamond_layer')
    # # The sides of the diamond are of length 2 * stride + 1
    # stride = 2

    # pos_connections = [] 
    # for x in range(0, cam_res, down_size):
    #     for y in range(0, cam_res, down_size):
    #         pos_connections += left_diag_connections(cam_res/2, x, y, stride, cam_res/2)

    # diamond_left = sim.Projection(left_diag_layer, diamond_layer, sim.FromListConnector(pos_connections), \
    #                               receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    # pos_connections = [] 
    # for x in range(0, cam_res, down_size):
    #     for y in range(0, cam_res, down_size):
    #         pos_connections += right_diag_connections(cam_res/2, x, y, stride, cam_res/2)

    # diamond_right = sim.Projection(right_diag_layer, diamond_layer, sim.FromListConnector(pos_connections), \
    #                                receptor_type='excitatory', synapse_type=sim.StaticSynapse(weight=exc_weight, delay=exc_delay))

    # diamond_layer.record(['spikes'])


    ##########################################################
    #### Run the simulation
    sim.run(sim_time)

    # neo = horizontal_layer.get_data(variables=['spikes'])
    # horizontal_spikes = neo.segments[0].spiketrains

    # neo = vertical_layer.get_data(variables=['spikes'])
    # vertical_spikes = neo.segments[0].spiketrains
    
    # neo = left_diag_layer.get_data(variables=['spikes'])
    # left_diag_spikes = neo.segments[0].spiketrains
    
    # neo = right_diag_layer.get_data(variables=['spikes'])
    # right_diag_spikes = neo.segments[0].spiketrains

    neo = square_layer.get_data(variables=['spikes'])
    square_spikes = neo.segments[0].spiketrains

    # neo = diamond_layer.get_data(variables=['spikes'])
    # diamond_spikes = neo.segments[0].spiketrains

    sim.end()


    ##########################################################
    #### Plot the receptive fields
    # line_properties = [{'color': 'red', 'markersize': 2}, {'color': 'blue', 'markersize': 2}]
    # plot.Figure(
    #     # plot.Panel(v, ylabel="Membrane potential (mV)", data_labels=[test_neuron.label], yticks=True, xlim=(0, sim_time)),
    #     # plot.Panel(pos_spikes, ylabel='Neuron idx', yticks=True, xticks=True, markersize=5, xlim=(0, sim_time)),#, \
    #     # xlim=(0, sim_time), line_properties=line_properties), 
    #     # plot spikes (or in this case spike)
    #     plot.Panel(horizontal_spikes, ylabel='Neuron idx', yticks=True, xlabel='Horizontal', xticks=True, markersize=2, xlim=(0, sim_time)), 
    #     plot.Panel(vertical_spikes, ylabel='Neuron idx', yticks=True, xlabel='Vertical', xticks=True, markersize=2, xlim=(0, sim_time)), 
    #     plot.Panel(left_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Left diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
    #     plot.Panel(right_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Right diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
    #     title='Receptive fields',
    #     annotations='Simulated with {}'.format(sim.name())
    # ) 
    # matplotlib.show()
    
    plot.Figure(
        plot.Panel(square_spikes, ylabel='Neuron idx', yticks=True, xlabel='Square shape', xticks=True, markersize=2, xlim=(0, sim_time)), 
        # plot.Panel(diamond_spikes, ylabel='Neuron idx', yticks=True, xlabel='Diamond shape', xticks=True, markersize=2, xlim=(0, sim_time)), 
        title='Shape detector',
        annotations='Simulated with {}'.format(sim.name())
    )
    matplotlib.show()

    # Process spiketrains for the square
    spiking_times = {}
    for neuron, spikes in enumerate(square_spikes):
        for spike in spikes:
            if not int(spike) in spiking_times:
                spiking_times[int(spike)] = []
            spiking_times[int(spike)].append(neuron)

    display_video(args.input, spiking_times, stride)


def display_video(filepath, spikes, stride):
    video_dev = cv2.VideoCapture(filepath)
    if not video_dev.isOpened():
        print('Video file could not be opened:', filepath)
        exit()

    fps = video_dev.get(cv2.CAP_PROP_FPS)
    frame_time_ms = int(1000./float(fps))
    height = int(video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_dev.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(video_dev.get(cv2.CAP_PROP_FRAME_COUNT))

    radius = stride // 2

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video_output = cv2.VideoWriter(filepath.strip('.txt') + '_result.avi', fourcc, float(fps), (width, height))

    for i in range(0, n_frames):
        read_correctly, frame = video_dev.read()
        if not read_correctly:
            break

        # Accumulate all spikes occurring between frames
        tmp = []
        for j in range(i*frame_time_ms, (i+1)*frame_time_ms):
            tmp.append(spikes.get(j))

        tmp = list(itertools.chain.from_iterable([k for k in tmp if k]))

        if tmp and len(tmp) > 0:
            spike = tmp[len(tmp)//2] # take median for now 
            row, col = coord_from_neuron(spike, height)
            cv2.rectangle(frame, (col-radius, row-radius), (col+radius, row+radius), (0, 0, 255), 1) 

        video_output.write(frame)
        # cv2.imshow('frame', frame)

        cv2.imwrite('output/frame_{0:05d}.png'.format(i),frame)

    video_dev.release()
    video_output.release()


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
