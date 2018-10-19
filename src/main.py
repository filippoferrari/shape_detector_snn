from __future__ import print_function

import argparse
import time

import cv2
import numpy as np
from numpy import int16, log2

import pydvs.generate_spikes as gs

from utils.constants import *

# -------------------------------------------------------------------- #
# grab / rescale frame                                                 #


def select_channel(frame, channel):
    if channel == RGB:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif channel == RED:
        return frame[:,:,0]
    elif channel == GREEN:
        return frame[:,:,0]
    elif channel == BLUE:
        return frame[:,:,0]


def grab_first(dev, res, channel):
    _, raw = dev.read()
    height, width, _ = raw.shape
    new_height = res
    new_width = int(float(new_height*width) / float(height))
    col_from = (new_width - res)//2
    col_to   = col_from + res
    img = cv2.resize(select_channel(raw, channel).astype(int16),
                     (new_width, new_height))[:, col_from:col_to]

    return img, new_width, new_height, col_from, col_to


def grab_frame(dev, width, height, col_from, col_to, channel):
    _, raw = dev.read()
    img = cv2.resize(select_channel(raw, channel).astype(int16),
                     (width, height))[:, col_from:col_to]

    return img


# ---------------------------------------------------------------------- #


def main(args):
    video_dev_id = args.video_id

    if len(video_dev_id) < 4:
        # Assume that urls have at least 4 characters
        video_dev_id = int(video_dev_id)

    print('Channel:', args.channel)
    print('Resolution:', args.res)
    print('Video id:', video_dev_id)

    mode = args.res
    cam_res = int(mode)
    width = cam_res  # square output
    height = cam_res
    shape = (height, width)
    channel = args.channel

    data_shift = uint8(log2(cam_res))
    up_down_shift = uint8(2*data_shift)
    data_mask = uint8(cam_res - 1)

    polarity = POLARITY_DICT[MERGED_POLARITY]
    output_type = OUTPUT_TIME
    history_weight = 1.0
    threshold = 12  # ~ 0.05*255
    max_threshold = 180  # 12*15 ~ 0.7*255

    scale_width = 0
    scale_height = 0
    col_from = 0
    col_to = 0

    curr     = np.zeros(shape,     dtype=int16)
    ref      = 128*np.ones(shape,  dtype=int16)
    spikes   = np.zeros(shape,     dtype=int16)
    diff     = np.zeros(shape,     dtype=int16)
    abs_diff = np.zeros(shape,     dtype=int16)

    # just to see things in a window
    spk_img  = np.zeros((height, width, 3), uint8)

    num_bits = 6   # how many bits are used to represent exceeded thresholds
    num_active_bits = 2  # how many of bits are active
    log2_table = gs.generate_log2_table(num_active_bits, num_bits)[num_active_bits - 1]
    spike_lists = None
    pos_spks = None
    neg_spks = None
    max_diff = 0

    # -------------------------------------------------------------------- #
    # inhibition related                                                   #

    inh_width = 2
    is_inh_on = True
    inh_coords = gs.generate_inh_coords(width, height, inh_width)

    # -------------------------------------------------------------------- #
    # camera/frequency related                                             #

    video_dev = cv2.VideoCapture(video_dev_id)  # webcam
    # video_dev = cv2.VideoCapture('/path/to/video/file')  # webcam

    print(video_dev.isOpened())

    # ps3 eyetoy can do 125fps
    try:
        video_dev.set(cv2.CAP_PROP_FPS, 125)
    except Exception:
        pass

    fps = video_dev.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 125.0
    max_time_ms = int(1000./float(fps))

    # ---------------------- main loop -------------------------------------#

    WINDOW_NAME = 'spikes'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    is_first_pass = True
    start_time = time.time()
    end_time = 0
    frame_count = 0
    while True:
        # get an image from video source
        if is_first_pass:
            curr[:], scale_width, scale_height, col_from, col_to = grab_first(video_dev, cam_res, channel)
            is_first_pass = False
        else:
            curr[:] = grab_frame(video_dev, scale_width,  scale_height, col_from, col_to, channel)

        # do the difference
        diff[:], abs_diff[:], spikes[:] = gs.thresholded_difference(curr, ref, threshold)

        # inhibition ( optional )
        if is_inh_on:
            spikes[:] = gs.local_inhibition(spikes, abs_diff, inh_coords,
                                            width, height, inh_width)

        # update the reference
        ref[:] = gs.update_reference_time_binary_thresh(abs_diff, spikes, ref,
                                                        threshold, max_time_ms,
                                                        num_active_bits,
                                                        history_weight,
                                                        log2_table)

        # convert into a set of packages to send out
        neg_spks, pos_spks, max_diff = gs.split_spikes(spikes, abs_diff, polarity)

        # this takes too long, could be parallelized at expense of memory
        spike_lists = gs.make_spike_lists_time_bin_thr(pos_spks, neg_spks,
                                                       max_diff,
                                                       up_down_shift, data_shift, data_mask,
                                                       max_time_ms,
                                                       threshold,
                                                       max_threshold,
                                                       num_bits,
                                                       log2_table)

        spk_img[:] = gs.render_frame(spikes, curr, cam_res, cam_res, polarity)
        cv2.imshow(WINDOW_NAME, spk_img.astype(uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()

        if end_time - start_time >= 1.0:
            print('{} frames per second'.format(frame_count))
            frame_count = 0
            start_time = time.time()
        else:
            frame_count += 1

    cv2.destroyAllWindows()
    cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--channel', default='RGB', required=False, type=str)
    parser.add_argument('-r', '--res', default=MODE_128, required=False, type=int)
    parser.add_argument('-v', '--video_id', default='0', required=False, type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
