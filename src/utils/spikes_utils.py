# -*- coding: utf-8 -*-

import numpy as np


def neuron_id(row, col, res):
    """ 
    Convert x,y pixel coordinate to neuron id coordinate
    """
    return row * res + col


def decode_spike(cam_res, key):
    """ 
    Decode DVS emulator output
    """
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
    """
    Create cube with spikes for visualisation purposes

    First dimension is the timestep
    Second is the row
    Third is the column
    """
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


def populate_spikes(raw_spikes, cam_res, sim_time):
    """
    Populate array to pass to a SpikeSourceArray
    
    Each line of the input is "[col][row][up|down], time_ms"

    Output is a list of times for each neuron
    [
        [t_01, t_02, t_03] Neuron 0 spikes times
        ... 
        [t_n1, t_n2, t_n3, t_n4] Neuron n spikes times
    ]
    """
    out_pos = []
    out_neg = []
    n_neurons = cam_res * cam_res

    for _ in range(n_neurons):
        out_pos.append(list())
        out_neg.append(list())

    for spike in raw_spikes:
        parts = spike.split(',')
        row, col, polarity = decode_spike(cam_res, int(parts[0]))
        spike_time = float(parts[1])
        if polarity:
            out_pos[neuron_id(row, col, cam_res)].append(spike_time)
        else:
            out_neg[neuron_id(row, col, cam_res)].append(spike_time)

    # out_pos_seq = [Sequence(i) for i in out_pos]
    # out_neg_seq = [Sequence(i) for i in out_neg]

    # return out_pos_seq, out_neg_seq

    return out_pos, out_neg

def read_recording_settings(args):
    """
    Read setting saved at the beginning of the file

    First line is DVS resolution - ALWAYS a square
    Second line is simulation time in milliseconds
    """
    print('Reading file...')
    with open(args.input, 'r') as fh:
        raw_spikes = fh.read().splitlines()
    print('    done')

    cam_res = int(raw_spikes[0])
    sim_time = int(raw_spikes[1])

    return raw_spikes[2:], cam_res, sim_time


def read_spikes_input(raw_spikes, cam_res, sim_time):
    """
    Read input file and parse informations
    """
    spikes_pos = []
    spikes_neg = []

    print('Resolution: {}'.format(cam_res))
    print('Simulation time: {} ms'.format(sim_time))

    print('Processing input file...')
    spikes_pos, spikes_neg = populate_spikes(raw_spikes, cam_res, sim_time)
    print('    done')
    print('')

    return spikes_pos, spikes_neg