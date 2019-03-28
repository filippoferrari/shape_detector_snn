# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np

import cv2

import pyNN.utility.plotting as plot

from src.dvs_emulator import DVS_Emulator

import spynnaker8 as sim
import spynnaker8.external_devices as ext

from src.utils.constants import OUTPUT_RATE, OUTPUT_TIME, OUTPUT_TIME_BIN_THR, KEY_SPINNAKER, KEY_XYP

from src.utils.debug_utils import receive_spikes, image_slice_viewer

from src.utils.spikes_utils import read_spikes_from_video, populate_debug_times_from_video, coord_from_neuron, \
                               read_recording_settings, read_spikes_input, neuron_id, populate_debug_times

from src.network_utils.receptive_fields import horizontal_connectivity_pos, horizontal_connectivity_neg, \
                                           vertical_connectivity_pos, vertical_connectivity_neg, \
                                           left_diagonal_connectivity_pos, left_diagonal_connectivity_neg, \
                                           right_diagonal_connectivity_pos, right_diagonal_connectivity_neg
from src.network_utils.shapes import hor_connections, vert_connections, left_diag_connections, right_diag_connections


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dont_save', action='store_true', default=False, help='Do not save results as video')
    parser.add_argument('-V', '--vis', action='store_true', default=False, help='Show visualisations')

    parser.add_argument('-i', '--input', required=False, type=str, help='Video file')
    parser.add_argument('-o', '--output_file', required=False, default=None, type=str, help='Save video and spikes DVS emulator')
    parser.add_argument('-v', '--video', required=False, action='store_true', default=False, help='Hardcode spikes from video')
    parser.add_argument('-w', '--webcam', required=False, default=False, action='store_true', help='Use webcam')

    args = parser.parse_args()

    return args


def main(args):
    # For some weird opencv/matplotlib bug, need to call matplotlib before opencv
    plt.plot([1,2,3])
    plt.close('all')

    if args.video:
        spikes_pos, spikes_neg, cam_res, sim_time = read_spikes_from_video(args.input)
    else:
        cam_res = 32
        if args.webcam:
            dvs = DVS_Emulator(cam_res, video_device='webcam', output_video=args.output_file)
        else:
            dvs = DVS_Emulator(cam_res, video_device=args.input, inhibition=False, output_type=OUTPUT_TIME)

        dvs.read_video_source()

        spikes_pos, spikes_neg = dvs.split_pos_neg_spikes()
        cam_res = dvs.cam_res
        sim_time = dvs.sim_time

        if args.output_file:
            dvs.save_output(args.output_file)

    #### Display input spikes
    if args.vis:
        if args.video:
            vis_spikes = populate_debug_times_from_video(spikes_pos, spikes_neg, cam_res, sim_time)
            image_slice_viewer(vis_spikes)
        else:
            image_slice_viewer(dvs.tuple_to_numpy(), step=dvs.time_bin_ms)


    n_total = cam_res * cam_res

    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 120)

    ##########################################################
    #### Some values for the network

    # Some values for the network 
    exc_weight = 3
    exc_delay = 1

    inh_weight = 1
    inh_delay = 1

    shapes_weight = 3 
    shapes_delay = 1

    down_size = 1

    ##########################################################
    #### Set the first layers of the network

    print(spikes_pos)

    # SpikeSourceArray for the positive polarity of the DVS
    stimulus_pos = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_pos), label='stimulus_pos')
    
    # SpikeSourceArray for the negative polarity of the DVS
    stimulus_neg = sim.Population(n_total, sim.SpikeSourceArray(spike_times=spikes_neg), label='stimulus_neg')


    ####################################################################################################################
    #### RECEPTIVE FIELDS
    ####################################################################################################################


    ##########################################################
    #### Horizontal receptive field
    horizontal_layer = sim.Population(n_total / (down_size * down_size), sim.IF_curr_exp(), label='horizontal_layer')

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
    vertical_layer = sim.Population(n_total / (down_size * down_size), sim.IF_curr_exp(), label='vertical_layer')

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
    left_diag_layer = sim.Population(n_total / (down_size * down_size), sim.IF_curr_exp(), label='left_diag_layer')

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
    right_diag_layer = sim.Population(n_total / (down_size * down_size), sim.IF_curr_exp(), label='right_diag_layer')

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
    square_layer = sim.Population(n_total / (down_size * down_size), sim.IF_curr_exp(), label='square_layer')
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
    for i in range(0, n_total / (down_size * down_size)):
        for j in range(0, n_total / (down_size * down_size)):
            if i != j:
                lateral_inh_connections.append((i, j))

    sim.Projection(square_layer, square_layer, sim.FromListConnector(lateral_inh_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inhibition_exc_w, delay=inhibition_delay))


    ##########################################################
    #### Diamond shape detector
    diamond_layer = sim.Population(n_total / (down_size * down_size), sim.IF_curr_exp(), label='diamond_layer')
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
    for i in range(0, n_total / (down_size * down_size)):
        for j in range(0, n_total / (down_size * down_size)):
            if i != j:
                lateral_inh_connections.append((i, j))

    sim.Projection(diamond_layer, diamond_layer, sim.FromListConnector(lateral_inh_connections), \
                   receptor_type='inhibitory', synapse_type=sim.StaticSynapse(weight=inhibition_exc_w, delay=inhibition_delay))


    # ##########################################################
    # #### Inhibition between shapes
    # shapes_inhibition = []
    # for i in range(0, n_total / (down_size * down_size)):
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
        plot.Panel(horizontal_spikes, ylabel='Neuron idx', yticks=True, xlabel='Horizontal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(vertical_spikes, ylabel='Neuron idx', yticks=True, xlabel='Vertical', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(left_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Left diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(right_diag_spikes, ylabel='Neuron idx', yticks=True, xlabel='Right diagonal', xticks=True, markersize=2, xlim=(0, sim_time)), 
        title='Receptive fields',
        annotations='Simulated with {}\n {}'.format(sim.name(), args.input)
    ) 
    plt.show()
    
    plot.Figure(
        plot.Panel(square_spikes, ylabel='Neuron idx', yticks=True, xlabel='Square shape', xticks=True, markersize=2, xlim=(0, sim_time)), 
        plot.Panel(diamond_spikes, ylabel='Neuron idx', yticks=True, xlabel='Diamond shape', xticks=True, markersize=2, xlim=(0, sim_time)), 
        title='Shape detector',
        annotations='Simulated with {}\n {}'.format(sim.name(), args.input)
    )
    plt.show()


    if not args.dont_save:
        # Process spiketrains for each shape
        spiking_times_square = shape_spikes_bin(square_spikes)
        spiking_times_diamond = shape_spikes_bin(diamond_spikes)

        if args.webcam:
            save_video(dvs.video_writer_path, [spiking_times_square,spiking_times_diamond], stride, ['r','y'])
        else:
            save_video(args.input, [spiking_times_square,spiking_times_diamond], stride, ['r','y'])


def shape_spikes_bin(shape_spikes):
    spiking_times = {}
    for neuron, spikes in enumerate(shape_spikes):
        for spike in spikes:
            if not int(spike) in spiking_times:
                spiking_times[int(spike)] = []
            spiking_times[int(spike)].append(neuron)
    return spiking_times


def save_video(filepath, list_of_spikes, stride, colours):

    # Colours in opencv are BGR
    colour = {'r':(0, 0, 255), 'g':(0, 255, 0), 'b':(255, 0, 0), 'y':(0, 255, 190)}

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

    filename = '{}_{}_{}'.format(filepath.strip('.txt').strip('avi'), datetime.datetime.now().isoformat(), 'result.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video_output = cv2.VideoWriter(filename, fourcc, float(fps), (width, height))

    for i in range(0, n_frames):
        read_correctly, frame = video_dev.read()
        if not read_correctly:
            break

        for index, spikes in enumerate(list_of_spikes):
            # Accumulate all spikes occurring between frames
            spikes_bin = []
            for j in range(i*frame_time_ms, (i+1)*frame_time_ms):
                spikes_bin.append(spikes.get(j))

            spikes_bin = sorted(list(itertools.chain.from_iterable([k for k in spikes_bin if k])))

            if spikes_bin and len(spikes_bin) > 0:
                spike = spikes_bin[len(spikes_bin)//2] # take median for now 
                row, col = coord_from_neuron(spike, height)
                cv2.rectangle(frame, (col-radius, row-radius), (col+radius, row+radius), colour[colours[index]], 1) 

        video_output.write(frame)
        # cv2.imshow('frame', frame)

        cv2.imwrite('output/frame_{0:05d}.png'.format(i),frame)

    video_dev.release()
    video_output.release()


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
