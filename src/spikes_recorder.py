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
    elif channel == VIDEO:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def grab_first(dev, res, channel):
    _, raw = dev.read()
    height, width, _ = raw.shape
    new_height = res
    new_width = int(float(new_height*width) / float(height))
    col_from = (new_width - res)//2
    col_to   = col_from + res
    img = cv2.resize(select_channel(raw, channel).astype(int16),
                     (new_width, new_height))[:, col_from:col_to]

    return img, new_width, new_height, col_from, col_to, raw


def grab_frame(raw, width, height, col_from, col_to, channel):
    img = cv2.resize(select_channel(raw, channel).astype(int16),
                    (width, height))[:, col_from:col_to]

    return img, raw

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
                      frame_time_ms, min_thresh, max_thresh, \
                      num_bins=1, log2_table=None):

    if output_type == OUTPUT_RATE:
        return gs.make_spike_lists_rate(pos, neg, max_diff,
                                     max_thresh,
                                     flag_shift, data_shift, data_mask,
                                     frame_time_ms,
                                     key_coding=KEY_SPINNAKER)
    elif output_type == OUTPUT_TIME_BIN_THR:
        return gs.make_spike_lists_time_bin_thr(pos, neg, max_diff,
                                                 flag_shift, data_shift, data_mask,
                                                 frame_time_ms,
                                                 min_thresh,
                                                 max_thresh,
                                                 num_bins,
                                                 log2_table,
                                                 key_coding=KEY_XYP)
    else:
        return gs.make_spike_lists_time(pos, neg, max_diff,
                                     flag_shift, data_shift, data_mask,
                                     num_bins,
                                     frame_time_ms,
                                     min_thresh,
                                     max_thresh,
                                     key_coding=KEY_XYP)


# ---------------------------------------------------------------------- #


def main(args):
    video_dev_id = args.video_id

    if len(video_dev_id) < 4:
        # Assume that urls have at least 4 characters
        video_dev_id = int(video_dev_id)

    cam_res = int(args.res)
    width = cam_res  # square output
    height = cam_res
    shape = (height, width)
    channel = args.channel

    data_shift = uint8(log2(cam_res))
    up_down_shift = uint8(2*data_shift)
    data_mask = uint8(cam_res - 1)

    polarity = POLARITY_DICT[MERGED_POLARITY]
    output_type = OUTPUT_TIME_BIN_THR
    history_weight = 1.0
    threshold = 12  # ~ 0.05*255
    max_threshold = 180  # 12*15 ~ 0.7*255

    scale_width = 0
    scale_height = 0
    col_from = 0
    col_to = 0

    print()
    print('Channel:', channel)
    print('Polarity:', polarity)
    print('Output Type:', output_type)
    print('Resolution:', cam_res)
    print('Video id:', video_dev_id)
    print()

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
    if args.input_video != 'webcam':
        video_dev = cv2.VideoCapture(args.input_video)
        channel = 'VIDEO'
        print('File opened correctly:', video_dev.isOpened())
    else:
        video_dev = cv2.VideoCapture(video_dev_id)  # webcam
        print('Webcam working:', video_dev.isOpened())

    if not video_dev.isOpened():
        print('Exiting because webcam/file is not working')
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

    if args.output_file and args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        video_writer = cv2.VideoWriter(args.output_file[:-4]+"_video.avi", fourcc, fps, (cam_res, cam_res))

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
            curr[:], scale_width, scale_height, col_from, col_to, frame = grab_first(video_dev, cam_res, channel)
            is_first_pass = False
        else:
            read_correctly, raw = video_dev.read()
            if not read_correctly:
                break
            curr[:], frame = grab_frame(raw, scale_width,  scale_height, col_from, col_to, channel)

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

        spike_lists = make_spikes_lists(output_type, 
                                        pos_spks, 
                                        neg_spks, 
                                        max_diff,
                                        up_down_shift, 
                                        data_shift, 
                                        data_mask,
                                        frame_time_ms,
                                        threshold,
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
        for spk_list in spike_lists:
            for spk in spk_list:
                output_spikes.append('{},{:f}'.format(spk, total_time + time_index))
            time_index += time_bin_ms
        
        total_time += frame_time_ms

        # write the frame
        if args.output_file and args.save_video:
            video_writer.write(cv2.resize(frame,(int(cam_res),int(cam_res))))



    if args.output_file:
        # First line is dimension of video
        with open(args.output_file, 'w') as fh:
            fh.write('{}\n'.format(cam_res))
            fh.write('{}\n'.format(total_time))
            fh.write('\n'.join(output_spikes))

        if args.save_video:
            video_writer.release()

    cv2.destroyAllWindows()
    cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--channel', default='RGB', required=False, type=str, help='Webcam channels to use [RGB, RED, GREEN, BLUE]')
    parser.add_argument('-i', '--input_video', default='webcam', required=False, type=str, help='Either \'webcam\' or the absolute path to the video file')
    parser.add_argument('-o', '--output_file', default=None, required=False, type=str, help='Absolute path for the output file')
    parser.add_argument('-r', '--res', default=MODE_128, required=False, type=int, help='Resolution, [16, 32, 64, 128, 256]')
    parser.add_argument('-v', '--video_id', default='0', required=False, type=str, help='Device to use, 0 is the integrated webcam')
    parser.add_argument('-V', '--save_video', action='store_false', default=True, help='Do not save the video file')

    args = parser.parse_args()

    if args.input_video != 'webcam':
        args.save_video = False

    return args


if __name__ == '__main__':
    args_parsed = parse_args()
    main(args_parsed)
