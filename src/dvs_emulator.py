# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import cv2
import numpy as np
from numpy import int16, log2

import pydvs.generate_spikes as gs

from src.utils.constants import *

from src.utils.spikes_utils import neuron_id

class DVS_Emulator():

    def __init__(self, cam_res, config,
                 video_device='webcam',
                 output_video=None, 
                 polarity=MERGED_POLARITY, output_type=OUTPUT_TIME_BIN_THR, 
                 inhibition=True, key_coding=KEY_XYP):

        self.cam_res = cam_res
        self.shape = (self.cam_res, self.cam_res)
        if config['input'] == 'webcam':
            self.channel = RGB
        else:
            self.channel = VIDEO
        self.sim_time = 0

        self.polarity = POLARITY_DICT[config['polarity']]
        self.output_type = config['output_type']
        self.inhibition = config['inhibition']
        self.key_coding = config['key_coding']

        self.video_device = config['input']
        self.output_video = config['output_file']
        self.video_writer_path = config['output_file']
        
        self.output_spikes_tuple = list()

        self.spikes = None

    def read_video_source(self):
        data_shift = uint8(log2(self.cam_res))
        up_down_shift = uint8(2*data_shift)
        data_mask = uint8(self.cam_res - 1)

        output_spikes = []

        # Default values from pyDVS
        history_weight = 1.0
        threshold = 12  # ~ 0.05*255
        max_threshold = 180  # 12*15 ~ 0.7*255

        curr     = np.zeros(self.shape,     dtype=int16)
        ref      = 128*np.ones(self.shape,  dtype=int16)
        spikes   = np.zeros(self.shape,     dtype=int16)
        diff     = np.zeros(self.shape,     dtype=int16)
        abs_diff = np.zeros(self.shape,     dtype=int16)

        # just to see things in a window
        spk_img  = np.zeros((self.cam_res, self.cam_res, 3), uint8)

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
        is_inh_on = self.inhibition
        inh_coords = gs.generate_inh_coords(self.cam_res, self.cam_res, inh_width)

        if self.video_device != 'webcam':
            video_dev = cv2.VideoCapture(self.video_device)
            self.channel = 'VIDEO'
            print('File opened correctly:', video_dev.isOpened())
        else:
            video_dev = cv2.VideoCapture(0)  # webcam
            print('Webcam working:', video_dev.isOpened())

        fps = video_dev.get(cv2.CAP_PROP_FPS)
        frame_time_ms = int(1000./float(fps))
        self.time_bin_ms = frame_time_ms // num_bits

        if self.output_video:
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            self.video_writer_path = self.output_video + '_video.avi'
            video_writer = cv2.VideoWriter(self.video_writer_path, fourcc, fps, self.shape)

        WINDOW_NAME = 'DVS Emulator'
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()

        # if self.output_type == OUTPUT_TIME or self.output_type == OUTPUT_RATE:
        #     num_bits = np.floor(frame_time_ms)
        # else:
        #     num_bits = 5.

        output_spikes = []
        output_spikes_tuple = []

        is_first_pass = True
        start_time = time.time()
        end_time = 0
        frame_count = 0

        while True:
            # get an image from video source
            if is_first_pass:
                curr[:], scale_width, scale_height, col_from, col_to, frame \
                    = self.grab_first(video_dev, self.cam_res, self.channel)
                is_first_pass = False
            else:
                read_correctly, raw = video_dev.read()
                if not read_correctly:
                    break
                curr[:], frame = self.grab_frame(raw, scale_width,  scale_height, col_from, col_to, self.channel)

            # do the difference
            diff[:], abs_diff[:], spikes[:] = gs.thresholded_difference(curr, ref, threshold)

            # inhibition ( optional )
            if is_inh_on:
                spikes[:] = gs.local_inhibition(spikes, abs_diff, inh_coords,\
                                                     self.cam_res, self.cam_res, inh_width)

            # update the reference
            ref[:] = self.update_ref(self.output_type, abs_diff, spikes, ref,
                                     threshold, frame_time_ms,
                                     num_bits,
                                     history_weight,
                                     log2_table)

            # convert into a set of packages to send out
            neg_spks, pos_spks, max_diff = gs.split_spikes(spikes, abs_diff, self.polarity)

            spike_lists = self.make_spikes_lists(self.output_type, 
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
                                                log2_table,
                                                key_coding=self.key_coding)

            spk_img[:] = gs.render_frame(spikes, curr, self.cam_res, self.cam_res, self.polarity)
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
                    output_spikes.append('{},{:f}'.format(spk, self.sim_time + time_index))
                    output_spikes_tuple.append((spk, self.sim_time + time_index))
                time_index += self.time_bin_ms
            
            self.sim_time += frame_time_ms

            # write the frame
            if self.output_video:
                video_writer.write(cv2.resize(frame,(int(self.cam_res),int(self.cam_res))))
        
        if self.output_video:
            video_writer.release()

        video_dev.release()

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        self.output_spikes = output_spikes[:]
        self.output_spikes_tuple = output_spikes_tuple[:]


    def select_channel(self, frame, channel):
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


    def grab_first(self, dev, res, channel):
        _, raw = dev.read()
        height, width, _ = raw.shape
        new_height = res
        new_width = int(float(new_height*width) / float(height))
        col_from = (new_width - res)//2
        col_to   = col_from + res
        img = cv2.resize(self.select_channel(raw, channel).astype(int16),
                        (new_width, new_height))[:, col_from:col_to]

        return img, new_width, new_height, col_from, col_to, raw


    def grab_frame(self, raw, width, height, col_from, col_to, channel):
        img = cv2.resize(self.select_channel(raw, channel).astype(int16),
                        (width, height))[:, col_from:col_to]

        return img, raw


    def update_ref(self, output_type, abs_diff, spikes, ref, thresh, frame_time_ms, \
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


    def make_spikes_lists(self, output_type, pos, neg, max_diff, \
                        flag_shift, data_shift, data_mask, \
                        frame_time_ms, min_thresh, max_thresh, \
                        num_bins=1, log2_table=None, key_coding=KEY_SPINNAKER):

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


    def tuple_to_numpy(self):
        spikes = np.zeros([self.sim_time, self.cam_res, self.cam_res])

        for spikes_time in self.output_spikes_tuple:
            # Split tuple
            spike = spikes_time[0]
            t = spikes_time[1]

            x = spike[0]
            y = spike[1]
            polarity = spike[2]

            if polarity == 1:
                spikes[t, x, y] = polarity
            elif polarity == 0:
                spikes[t, x, y] = -1
        return spikes


    def split_pos_neg_spikes(self):
        out_pos = []
        out_neg = []
        n_neurons = self.cam_res * self.cam_res

        for _ in range(n_neurons):
            out_pos.append(list())
            out_neg.append(list())

        for spikes_time in self.output_spikes_tuple:
            # Split tuple
            spike = spikes_time[0]
            t = spikes_time[1]

            x = spike[0]
            y = spike[1]
            polarity = spike[2]

            if polarity:
                out_pos[neuron_id(x, y, self.cam_res)].append(t)
            else:
                out_neg[neuron_id(x, y, self.cam_res)].append(t)

        return out_pos, out_neg


    def save_output(self, output):
        with open(output + '_spikes.txt', 'w') as fh:
            fh.write('{}\n'.format(self.cam_res))
            fh.write('{}\n'.format(self.sim_time))
            fh.write('\n'.join(self.output_spikes))
