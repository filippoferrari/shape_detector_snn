# -*- coding: utf-8 -*-

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


def update_ref(output_type, abs_diff, spikes, ref, thresh, frame_time_ms, \
               num_spikes=1, history_weight=1., log2_table=None):
    
    if output_type == OUTPUT_RATE:
        return gs.update_reference_rate(abs_diff, spikes, ref, thresh,
                                     frame_time_ms,
                                     history_weight)

    elif output_type == OUTPUT_TIME_BIN_THR:
        
        return gs.update_reference_time_binary_thresh(abs_diff, spikes, ref,
                                                   thresh,
                                                   frame_time_ms,
                                                   num_spikes=num_spikes,
                                                   history_weight=history_weight,
                                                   log2_table=log2_table)
    else:
        return gs.update_reference_time_thresh(abs_diff, spikes, ref,
                                            thresh,
                                            frame_time_ms,
                                            history_weight)


def make_spikes_lists(output_type, pos, neg, max_diff, \
                      flag_shift, data_shift, data_mask, \
                      frame_time_ms, thresh, \
                      num_bins=1, log2_table=None):

    if output_type == OUTPUT_RATE:
        spin = gs.make_spike_lists_rate(pos, neg, max_diff,
                                     thresh,
                                     flag_shift, data_shift, data_mask,
                                     frame_time_ms,
                                     key_coding=KEY_SPINNAKER)

        xyp = make_spike_lists_rate(pos, neg, max_diff,
                                     thresh,
                                     flag_shift, data_shift, data_mask,
                                     frame_time_ms,
                                     key_coding=KEY_SPINNAKER)
        return spin, xyp
    elif output_type == OUTPUT_TIME_BIN_THR:
        return gs.make_spike_lists_time_bin_thr(pos, neg, max_diff,
                                                 flag_shift, data_shift, data_mask,
                                                 frame_time_ms,
                                                 thresh,
                                                 thresh,
                                                 num_bins,
                                                 log2_table,
                                                 key_coding=KEY_XYP)
    else:
        return gs.make_spike_lists_time(pos, neg, max_diff,
                                     flag_shift, data_shift, data_mask,
                                     frame_time_ms,
                                     frame_time_ms,
                                     thresh,
                                     thresh,
                                     key_coding=KEY_XYP)

########################


def grab_spike_key( row,  col, flag_shift,  data_shift, data_mask, is_pos_spike,  key_coding=KEY_SPINNAKER):
    spike_key = spike_to_xyp(row, col, is_pos_spike)
                             
    return spike_key

def spike_to_xyp( row,  col, is_pos_spike):
  return np.array([col, row, is_pos_spike])

def make_spike_lists_rate(pos_spikes,
                          neg_spikes,
                          global_max,
                          threshold,
                          flag_shift,
                          data_shift,
                          data_mask,
                          max_time_ms,
                          key_coding=KEY_SPINNAKER):
    """
        Convert spike (row, col, val, sign) lists into a list of Address
        Event Representation (AER) encoded spikes. Rate-encoded values.
        :param pos_spikes:  Positive (up) spikes to encode
        :param neg_spikes:  Negative (down) spikes to encode
        :param global_max:  Maximum change that happened for current frame,
                            used to limit the number of memory slots needed
        :param flag_shift:  How many bits to shift for the pos/neg bit (depends on resolution)
        :param data_shift:  How many bits to shift for the row (depends on resolution)
        :param data_mask:   Bits to take into account for the row/column information
        :param max_time_ms: Upper limit to the number of spikes that can be sent out
        :returns list_of_lists: A list containing lists of keys that should be sent. Each
                                key in the internal lists should be sent "at the same time"
    """
    max_spikes = max_time_ms
    len_neg = len(neg_spikes[0])
    len_pos = len(pos_spikes[0])
    max_pix = len_neg + len_pos

    list_of_lists = list()

    for list_idx in range(max_spikes):
        list_of_lists.append( list() )

    for pix_idx in range(max_pix):
        spike_key = 0

        if pix_idx < len_pos:
            spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx], \
                                        pos_spikes[COLS, pix_idx], \
                                        flag_shift, data_shift, data_mask,\
                                        is_pos_spike = 1,
                                        key_coding=key_coding)

            val = pos_spikes[VALS, pix_idx]//threshold
            val = (max_spikes - 1) - val
            spike_idx = max(0, val)

        else:
            neg_idx = pix_idx - len_pos
            spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx], \
                                        neg_spikes[COLS, neg_idx], \
                                        flag_shift, data_shift, data_mask,\
                                        is_pos_spike = 0,
                                        key_coding=key_coding)

            val = neg_spikes[VALS, neg_idx]//threshold
            val = (max_spikes - 1) - val
        #~       print("neg rate spikes val, key", val, spike_key)
            spike_idx = max(0, val)
        
        for list_idx in range(spike_idx):
            list_of_lists[list_idx].append(spike_key)



    return list_of_lists

# ---------------------------------------------------------------------- #


def main(args):
    video_dev_id = args.video_id

    if len(video_dev_id) < 4:
        # Assume that urls have at least 4 characters
        video_dev_id = int(video_dev_id)

    print('Channel:', args.channel)
    print('Resolution:', args.res)
    print('Video id:', video_dev_id)

    cam_res = int(args.res)
    width = cam_res  # square output
    height = cam_res
    shape = (height, width)
    channel = args.channel

    data_shift = uint8(log2(cam_res))
    up_down_shift = uint8(2*data_shift)
    data_mask = uint8(cam_res - 1)

    polarity = POLARITY_DICT[MERGED_POLARITY]
    output_type = OUTPUT_RATE
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

    print('Webcam working:', video_dev.isOpened())

    if not video_dev.isOpened():
        print('Exiting because webcam is not working')
        exit()

    # ps3 eyetoy can do 125fps
    try:
        video_dev.set(cv2.CAP_PROP_FPS, 125)
    except Exception:
        pass

    fps = video_dev.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 125.0
    frame_time_ms = int(1000./float(fps))
    time_bin_ms = frame_time_ms // num_bits

    # ---------------------- main loop -------------------------------------#

    WINDOW_NAME = 'spikes'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    output_spikes = []

    # if output_type == OUTPUT_TIME or output_type == OUTPUT_RATE:
    #     num_bits = np.floor(frame_time_ms)
    # else:
    #     num_bits = 5.

    is_first_pass = True
    start_time = time.time()
    end_time = 0
    frame_count = 0
    total_time = 0

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
        ref[:] = update_ref(output_type, abs_diff, spikes, ref,
                            threshold, frame_time_ms,
                            num_bits,
                            history_weight,
                            log2_table)

        # convert into a set of packages to send out
        neg_spks, pos_spks, max_diff = gs.split_spikes(spikes, abs_diff, polarity)

        spike_lists, xyp_lists = make_spikes_lists(output_type, 
                                        pos_spks, 
                                        neg_spks, 
                                        max_diff,
                                        up_down_shift, 
                                        data_shift, 
                                        data_mask,
                                        frame_time_ms,
                                        max_threshold,
                                        num_bits, 
                                        log2_table)

        spk_img[:] = gs.render_frame(spikes, curr, cam_res, cam_res, polarity)
        cv2.imshow(WINDOW_NAME, spk_img.astype(uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()

        # Compute frames per second
        if end_time - start_time >= 1.0:
            print('{} frames per second'.format(frame_count))
            frame_count = 0
            start_time = time.time()
        else:
            frame_count += 1

        # Write spikes out in correct format
        time_index = 0
        for spk_list, xyp_list in zip(spike_lists, xyp_lists):
            for spk, xyp in zip(spk_list, xyp_list):
                output_spikes.append('{},{:f},{}'.format(spk, total_time + time_index, xyp))
            time_index += time_bin_ms
        
        total_time += frame_time_ms

    if args.output_file:
        # First line is dimension of video
        with open(args.output_file, 'w') as fh:
            fh.write('{}\n'.format(cam_res))
            fh.write('{}\n'.format(total_time))
            fh.write('\n'.join(output_spikes))

    cv2.destroyAllWindows()
    cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--channel', default='RGB', required=False, type=str, help='Webcam channels to use [RGB, RED, GREEN, BLUE]')
    parser.add_argument('-o', '--output_file', default=None, required=False, type=str, help='Absolute path for the output file')
    parser.add_argument('-r', '--res', default=MODE_128, required=False, type=int, help='Resolution, [16, 32, 64, 128, 256]')
    parser.add_argument('-v', '--video_id', default='0', required=False, type=str, help='Device to use, 0 is the integrated webcam')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
