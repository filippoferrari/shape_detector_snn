# -*- coding: utf-8 -*-

import argparse
import datetime
import itertools
import os
import yaml

import cv2

from src.utils.spikes_utils import coord_from_neuron


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dont_save', action='store_true', default=False, help='Do not save results as video')
    parser.add_argument('-V', '--vis', action='store_true', default=False, help='Show visualisations')

    parser.add_argument('-c', '--config', required=True, type=str, help='Config file')
    parser.add_argument('-i', '--input', required=False, type=str, help='Video file')
    parser.add_argument('-o', '--output_file', required=False, default=None, type=str, help='Save video and spikes DVS emulator')
    parser.add_argument('-v', '--video', required=False, action='store_true', default=False, help='Hardcode spikes from video')
    parser.add_argument('-w', '--webcam', required=False, action='store_true', default=False, help='Use webcam')

    args = parser.parse_args()

    return args


def set_key(config, key, default=None):
    if key in config:
        return config[key]
    else:
        return default


def read_config(args):
    with open(os.path.expanduser(args.config), 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    if args.input:
        config['input'] = args.input
    else:
        config['input'] = set_key(config, 'input', default=None)

    if args.output_file:
        config['output_file'] = args.output_file
    else: 
        config['output_file'] = set_key(config, 'output_file', default=None)

    if args.video:
        config['video'] = args.video
    else:
        config['video'] = set_key(config, 'video', default=False)

    if args.webcam:
        config['webcam'] = args.webcam
    else:
        config['webcam'] = set_key(config, 'webcam', default=False)

    if args.dont_save:
        config['dont_save'] = args.dont_save
    else:
        config['dont_save'] = set_key(config, 'dont_save', default=False)

    if args.vis:
        config['vis'] = args.vis
    else:
        config['vis'] = set_key(config, 'vis', default=False)

    return config


def save_video(config, filepath, list_of_spikes, stride, colours):

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

    filename = '{}_{}_{}_{}_{}'.format(filepath.strip('.txt').strip('avi'), config['output_type'],\
                                       config['video'], datetime.datetime.now().isoformat(), 'result.avi')
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
                x, y = coord_from_neuron(spike, height)
                cv2.rectangle(frame, (x-radius, y-radius), (x+radius, y+radius), colour[colours[index]], 1) 

        video_output.write(frame)
        # cv2.imshow('frame', frame)

        cv2.imwrite('output/frame_{0:05d}.png'.format(i),frame)

    video_dev.release()
    video_output.release()