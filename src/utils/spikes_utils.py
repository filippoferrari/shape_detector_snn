# -*- coding: utf-8 -*-

import numpy as np

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


# Debug method to visualise the spikes
def populate_debug_times(raw_spikes, cam_res, sim_time):
    out = np.zeros([sim_time, cam_res, cam_res])

    for spike in raw_spikes:
        # Format of each line is "neuron_id time_ms"
        parts = spike.split(',')
        row, col, polarity = decode_spike(cam_res, int(parts[0]))
        spike_time = int(float(parts[1]))
        if spike_time >= sim_time:
            continue
        if polarity == 1:
            out[spike_time, row, col] = polarity
        elif polarity == 0:
            out[spike_time, row, col] = -1

    return out


# Populate array to pass to the SpikeSourceArray
def populate_spikes(raw_spikes, cam_res, sim_time):
    out_pos = []
    out_neg = []
    n_neurons = cam_res * cam_res

    for _ in range(n_neurons):
        out_pos.append(list())
        out_neg.append(list())

    for spike in raw_spikes:
        # Format of each line is "[col][row][up|down] time_ms"
        parts = spike.split(',')
        row, col, polarity = decode_spike(cam_res, int(parts[0]))
        spike_time = float(parts[1])
        if polarity:
            out_pos[row*cam_res+col].append(spike_time)
        else:
            out_neg[row*cam_res+col].append(spike_time)

    return out_pos, out_neg


# Read setting saved at the beginning of the file
def read_recording_settings(args):
    print('Reading file...')
    with open(args.input, 'r') as fh:
        raw_spikes = fh.read().splitlines()
    print('    done')

    cam_res = int(raw_spikes[0])
    sim_time = int(raw_spikes[1])

    return raw_spikes[2:], cam_res, sim_time


# Read input file and parse informations
def read_spikes_input(raw_spikes, cam_res, sim_time):
    spikes_pos = []
    spikes_neg = []

    print('Resolution: {}'.format(cam_res))
    print('Simulation time: {} ms'.format(sim_time))

    print('Processing input file...')
    spikes_pos, spikes_neg = populate_spikes(raw_spikes, cam_res, sim_time)
    print('    done')
    print('')

    return spikes_pos, spikes_neg