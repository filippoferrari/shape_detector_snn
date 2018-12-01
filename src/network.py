# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import numpy as np

import spynnaker8 as sim
import spynnaker8.external_devices as ext


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True, type=str, help='Text file with the spikes')

    args = parser.parse_args()

    return args


def decode_spike(cam_res, key):
    # Resolution at which the data is encoded
    data_shift = np.uint8(np.log2(cam_res))

    # Format is [col][row][up|down] where 
    # [col] and [row] are data_shift long
    # [up|down] is one bit 
    col = key >> (data_shift + 1)
    row = (key >> 1) & ((0b1 << data_shift) -1)
    polarity = key & 1

    return row, col, polarity


# Populate array to pass to the SpikeSourceArray
def populate_spikes(cam_res, lines):
    out = []
    n_neurons = cam_res * cam_res

    for _ in range(n_neurons):
        out.append(list())

    for line in lines:
        # Format of each line is "neuron_id time_ms"
        parts = line.split(',')
        row, col, polarity = decode_spike(cam_res, int(parts[0]))
        temp = filter(None, parts[2].strip('[').strip(']').split(' '))
        if int(temp[0]) != col or int(temp[1]) != row or int(temp[2]) != polarity:
            print('{} {} - {} {} - {} {}'.format(temp[0],col, temp[1],row, temp[2],polarity))

    return out 


# Read input file and parse informations
def read_spikes_input(filepath):
    spikes = []

    print('Reading file...')
    with open(filepath, 'r') as fh:
        lines = fh.read().splitlines()
    print('    done')

    cam_res = int(lines[0])
    sim_time = int(lines[1])

    print('Resolution:', cam_res)
    print('Simulation time:', sim_time)

    spikes = populate_spikes(cam_res, lines[2:])

    return cam_res, sim_time, spikes


# Print a message when each neuron in the stimulus layer spikes
def receive_spikes(label, time, neuron_ids):
    for neuron_id in neuron_ids:
        print("Neuron id: {} - Time: {} - Label: {}".format(neuron_id, time, label))


def main(args):
    # Read the input file
    cam_res, sim_time, spikes = read_spikes_input(args.input)

    exit()

    n_total = cam_res * cam_res

    sim.setup(timestep=1.0)

    # Set the first layer of the network
    spikeArray = {'spike_times': spikes}
    stimulus = sim.Population(n_total, sim.SpikeSourceArray, spikeArray, label='stimulus')

    # Activate live output from the first layer - testing only
    ext.activate_live_output_for(stimulus, tag=1, port=17897)
    live_spikes_connection = sim.external_devices.SpynnakerLiveSpikesConnection(receive_labels=["stimulus"])
    live_spikes_connection.add_receive_callback("stimulus", receive_spikes)

    sim.run(sim_time)

    sim.end()


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
