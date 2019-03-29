# -*- coding: utf-8 -*-
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from src.utils.spikes_utils import populate_debug_times, read_recording_settings

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True, type=str, help='Text file with the spikes')

    args = parser.parse_args()

    return args


def receive_spikes(label, time, neuron_ids):
    """
    Print a message when each neuron in the stimulus layer spikes
    """
    for neuron_id in neuron_ids:
        print("Neuron id: {} - Time: {} - Label: {}".format(neuron_id, time, label))


def image_slice_viewer(cube, step=41):
    class IndexTracker(object):
        def __init__(self, ax, X, step=41):
            self.ax = ax
            ax.figure.subplots_adjust(left=0.25, bottom=0.25)
            ax.set_title('use scroll wheel to navigate images')

            self.step = step
            self.X = X
            self.slices, rows, cols = X.shape
            self.ind = 0

            self.im = ax.imshow(self.X[self.ind, :, :], vmin=np.min(X), vmax=np.max(X))

            ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
            self.slider = Slider(ax, 'Axis %i index' % self.slices, 0, self.slices,
                            valinit=self.ind, valfmt='%i')
            self.slider.on_changed(self.update_slider)

            self.update()

        def press(self, event):
            if event.key == 'right':
                self.ind = (self.ind + self.step) % self.slices
            elif event.key == 'left':
                self.ind = (self.ind - self.step) % self.slices
            self.slider.set_val(self.ind)
            self.update()

        def update_slider(self, event):
            ind = int(self.slider.val)
            self.ind = ind
            self.update()

        def update(self):
            self.im.set_data(self.X[self.ind, :, :].T)
            ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, cube, step=step)

    fig.canvas.mpl_connect('key_press_event', tracker.press)
    plt.show()


def main(args):
    # Read the input file
    raw_spikes, cam_res, sim_time = read_recording_settings(args)
    times_debug = populate_debug_times(raw_spikes, cam_res, sim_time)
    image_slice_viewer(times_debug)

if __name__ == '__main__':
    args = parse_args()
    main(args)