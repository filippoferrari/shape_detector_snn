# -*- coding: utf-8 -*-

import argparse
import os
import yaml


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