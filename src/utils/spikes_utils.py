# -*- coding: utf-8 -*-

import cv2
import numpy as np


def neuron_id(row, col, res):
    """ 
    Convert x,y pixel coordinate to neuron id coordinate
    """
    if check_bounds(row, col, res):
        return row * res + col
    else:
        return []


def coord_from_neuron(neuron, res):
    """ 
    Convert neuron id to neuron x,y pixel coordinate
    """
    col = neuron % res
    row = (neuron - col) // res
    return row, col


def check_bounds(x, y, res):
    """
    Check if (x,y) is inside the square res x res
    """
    return x >= 0 and x < res and y >= 0 and y < res


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


def read_spikes_from_video(filepath):
    """
    Read video file as input, instead of spikes
    
    Output is a list of times for each neuron
    [
        [t_01, t_02, t_03] Neuron 0 spikes times
        ... 
        [t_n1, t_n2, t_n3, t_n4] Neuron n spikes times
    ]
    """
    video_dev = cv2.VideoCapture(filepath)

    if not video_dev.isOpened():
        print('Video file could not be opened:', filepath)
        exit()

    # Get video settings
    fps = video_dev.get(cv2.CAP_PROP_FPS)
    frame_time_ms = int(1000./float(fps))
    height = int(video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_dev.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(video_dev.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video needs to be a square
    if height != width:
        print('Width: {} - Height: {} - Not a square'.format(width, height))
        video_dev.release()
        exit()

    res = width
    n_neurons = res * res
    sim_time = n_frames * frame_time_ms

    # Initialise spikes lists
    pos_spikes = []
    neg_spikes = []
    for _ in range(n_neurons):
        pos_spikes.append(list())
        neg_spikes.append(list())

    
    # For each frame
    previous_frame = None
    for i in range(0, n_frames):
        read_correctly, frame = video_dev.read()
        if not read_correctly:
            break
        
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # If a pixel is not zero, the corresponding neuron will spike
        for row in range(0, frame.shape[0]):
            for column in range(0, frame.shape[1]):
                if frame[row,column] > 128:
                    pos_spikes[neuron_id(row,column,res)].append(i * frame_time_ms)
                if previous_frame is not None and previous_frame[row,column] > 128 and frame[row,column] < 128:
                    neg_spikes[neuron_id(row,column,res)].append(i * frame_time_ms)

                # if previous_frame is not None:
                #     # Current pixel moved from dark to bright
                #     if frame[row,column] > 128 and previous_frame[row,column] < 128:
                #         pos_spikes[neuron_id(row,column,res)].append(i * frame_time_ms)
                #     # Current pixel went from bright to dark
                #     elif previous_frame[row,column] > 128 and frame[row,column] < 128:
                #         neg_spikes[neuron_id(row,column,res)].append(i * frame_time_ms)
                # elif:
                #     if frame[row,column] > 128:
                #         pos_spikes[neuron_id(row,column,res)].append(i * frame_time_ms)

        if i > 0:
            previous_frame = frame.copy()

    return pos_spikes, neg_spikes, res, sim_time


def populate_debug_times_from_video(spikes, cam_res, sim_time):
    out = np.zeros([sim_time, cam_res, cam_res])

    for i, neuron in enumerate(spikes):
        for spike in neuron:
            row, col = coord_from_neuron(i, cam_res)
            out[spike, row, col] = 1
    
    return out