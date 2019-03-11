# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import cv2
import numpy as np
from numpy import int16, log2

import pydvs.generate_spikes as gs

from utils.constants import *


class DVS_Emulator():

    def __init__(self, cam_res, 
                 video_device='webcam',
                 output_video=None, 
                 polarity=MERGED_POLARITY, output_type=OUTPUT_TIME_BIN_THR, 
                 inhibition=True):
        self.cam_res = cam_res
        self.shape = (self.cam_res, self.cam_res)
        self.channel = RGB

        self.data_shift = uint8(log2(self.cam_res))
        self.up_down_shift = uint8(2*self.data_shift)
        self.data_mask = uint8(self.cam_res - 1)

        self.polarity = POLARITY_DICT[polarity]
        self.output_type = output_type

        self.output_spikes = []
        self.sim_time = 0

        # Default values from pyDVS
        self.history_weight = 1.0
        self.threshold = 12  # ~ 0.05*255
        self.max_threshold = 180  # 12*15 ~ 0.7*255

        self.curr     = np.zeros(self.shape,     dtype=int16)
        self.ref      = 128*np.ones(self.shape,  dtype=int16)
        self.spikes   = np.zeros(self.shape,     dtype=int16)
        self.diff     = np.zeros(self.shape,     dtype=int16)
        self.abs_diff = np.zeros(self.shape,     dtype=int16)

        # just to see things in a window
        self.spk_img  = np.zeros((self.cam_res, self.cam_res, 3), uint8)

        self.num_bits = 6   # how many bits are used to represent exceeded thresholds
        self.num_active_bits = 2  # how many of bits are active
        self.log2_table = gs.generate_log2_table(self.num_active_bits, self.num_bits)[self.num_active_bits - 1]
        self.spike_lists = None
        self.pos_spks = None
        self.neg_spks = None
        self.max_diff = 0

        # -------------------------------------------------------------------- #
        # inhibition related                                                   #

        self.inh_width = 2
        self.is_inh_on = inhibition
        self.inh_coords = gs.generate_inh_coords(self.cam_res, self.cam_res, self.inh_width)

        if video_device != 'webcam':
            self.video_dev = cv2.VideoCapture(video_device)
            self.channel = 'VIDEO'
            print('File opened correctly:', self.video_dev.isOpened())
        else:
            self.video_dev = cv2.VideoCapture(0)  # webcam
            print('Webcam working:', self.video_dev.isOpened())

        self.fps = self.video_dev.get(cv2.CAP_PROP_FPS)
        self.frame_time_ms = int(1000./float(self.fps))
        self.time_bin_ms = self.frame_time_ms // self.num_bits

        self.output_video = output_video
        if self.output_video:
            self.fourcc = cv2.VideoWriter_fourcc(*'MP42')
            self.video_writer_path = output_video + '_video.avi'
            self.video_writer = cv2.VideoWriter(self.video_writer_path,\
                                                self.fourcc, self.fps, self.shape)

    def read_video_source(self):

        WINDOW_NAME = 'DVS Emulator'
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()

        # if output_type == OUTPUT_TIME or output_type == OUTPUT_RATE:
        #     num_bits = np.floor(frame_time_ms)
        # else:
        #     num_bits = 5.

        output_spikes = []

        is_first_pass = True
        start_time = time.time()
        end_time = 0
        frame_count = 0

        while True:
            # get an image from video source
            if is_first_pass:
                self.curr[:], scale_width, scale_height, col_from, col_to, frame \
                    = self.grab_first(self.video_dev, self.cam_res, self.channel)
                is_first_pass = False
            else:
                read_correctly, raw = self.video_dev.read()
                if not read_correctly:
                    break
                self.curr[:], frame = self.grab_frame(raw, scale_width,  scale_height, col_from, col_to, self.channel)

            # do the difference
            self.diff[:], self.abs_diff[:], self.spikes[:] = gs.thresholded_difference(self.curr, self.ref, self.threshold)

            # inhibition ( optional )
            if self.is_inh_on:
                self.spikes[:] = gs.local_inhibition(self.spikes, self.abs_diff, self.inh_coords,
                                                     self.cam_res, self.cam_res, self.inh_width)

            # update the reference
            self.ref[:] = self.update_ref(self.output_type, self.abs_diff, self.spikes, self.ref,
                                          self.threshold, self.frame_time_ms,
                                          self.num_bits,
                                          self.history_weight,
                                          self.log2_table)

            # convert into a set of packages to send out
            neg_spks, pos_spks, max_diff = gs.split_spikes(self.spikes, self.abs_diff, self.polarity)

            spike_lists = self.make_spikes_lists(self.output_type, 
                                                pos_spks, 
                                                neg_spks, 
                                                max_diff,
                                                self.up_down_shift, 
                                                self.data_shift, 
                                                self.data_mask,
                                                self.frame_time_ms,
                                                self.max_threshold,
                                                self.num_bits, 
                                                self.log2_table)

            self.spk_img[:] = gs.render_frame(self.spikes, self.curr, self.cam_res, self.cam_res, self.polarity)
            cv2.imshow(WINDOW_NAME, self.spk_img.astype(uint8))

            if cv2.waitKey(1) & 0xFF == ord('q') or self.sim_time > 5000:
                break

            end_time = time.time()

            # Compute frames per second
            if end_time - start_time >= 1.0:
                print('{} frames per second'.format(frame_count))
                frame_count = 0
                start_time = time.time()
            else:
                frame_count += 1

            print(spike_lists)



            # Write spikes out in correct format
            time_index = 0
            for spk_list in spike_lists:
                for spk in spk_list:
                    output_spikes.append('{},{:f}'.format(spk, self.sim_time + time_index))
                time_index += self.time_bin_ms
            
            self.sim_time += self.frame_time_ms

            # write the frame
            if self.output_video:
                self.video_writer.write(cv2.resize(frame,(int(self.cam_res),int(self.cam_res))))
        
        if self.output_video:
            self.video_writer.release()

        self.video_dev.release()

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        print(output_spikes)

        self.output_spikes = output_spikes[:]

        print(self.output_spikes)


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
                        frame_time_ms, thresh, \
                        num_bins=1, log2_table=None):

        if output_type == OUTPUT_RATE:
            return gs.make_spike_lists_rate(pos, neg, max_diff,
                                        thresh,
                                        flag_shift, data_shift, data_mask,
                                        frame_time_ms,
                                        key_coding=KEY_SPINNAKER)
        elif output_type == OUTPUT_TIME_BIN_THR:
            return gs.make_spike_lists_time_bin_thr(pos, neg, max_diff,
                                                    flag_shift, data_shift, data_mask,
                                                    frame_time_ms,
                                                    thresh,
                                                    thresh,
                                                    num_bins,
                                                    log2_table,
                                                    key_coding=KEY_SPINNAKER)
        else:
            return gs.make_spike_lists_time(pos, neg, max_diff,
                                        flag_shift, data_shift, data_mask,
                                        frame_time_ms,
                                        frame_time_ms,
                                        thresh,
                                        thresh,
                                        key_coding=KEY_SPINNAKER)
