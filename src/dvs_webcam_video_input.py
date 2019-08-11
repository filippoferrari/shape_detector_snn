# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import cv2

import pyNN.utility.plotting as plot

from src.dvs_emulator import DVS_Emulator

from src.utils.constants import OUTPUT_RATE, OUTPUT_TIME, OUTPUT_TIME_BIN_THR, KEY_SPINNAKER, KEY_XYP

from src.utils.debug_utils import receive_spikes, image_slice_viewer

from src.utils.io_utils import parse_args, read_config

from src.utils.spikes_utils import read_spikes_from_video, populate_debug_times_from_video, coord_from_neuron, \
                               read_recording_settings, neuron_id, populate_debug_times

from src.network_utils.receptive_fields import horizontal_connectivity_pos, horizontal_connectivity_neg, \
                                           vertical_connectivity_pos, vertical_connectivity_neg, \
                                           left_diagonal_connectivity_pos, left_diagonal_connectivity_neg, \
                                           right_diagonal_connectivity_pos, right_diagonal_connectivity_neg
from src.network_utils.shapes import hor_connections, vert_connections, left_diag_connections, right_diag_connections


def main(config):
    # For some weird opencv/matplotlib bug, need to call matplotlib before opencv
    plt.plot([1,2,3])
    plt.close('all')

    if config['video']:
        spikes_pos, spikes_neg, cam_res, sim_time = read_spikes_from_video(config['input'])
    else:
        cam_res = 32
        dvs = DVS_Emulator(cam_res, config)

        dvs.read_video_source()

        spikes_pos, spikes_neg = dvs.split_pos_neg_spikes()
        cam_res = dvs.cam_res
        sim_time = dvs.sim_time

        if config['output_file']:
            dvs.save_output(config['output_file'])

    #### Display input spikes
    if config['vis']:
        if config['video']:
            vis_spikes = populate_debug_times_from_video(spikes_pos, spikes_neg, cam_res, sim_time)
            image_slice_viewer(vis_spikes)
        else:
            image_slice_viewer(dvs.tuple_to_numpy(), step=dvs.time_bin_ms)


if __name__ == '__main__':
    args_parsed = parse_args()
    config = read_config(args_parsed)
    main(config)
