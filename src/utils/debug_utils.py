# -*- coding: utf-8 -*-
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from spikes_utils import populate_debug_times, read_recording_settings

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


def cube_show_slider(cube, axis=0, **kwargs):
    """
    Visualise the spikes, each frame is a timestep

    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """

    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")

    # generate figure
    fig = plt.figure()
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    s = [slice(1,2) if i == axis else slice(None) for i in xrange(3)]
    im = cube[s].squeeze()

    # display image
    l = ax.imshow(im, vmin=-1, vmax=1, **kwargs)

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None) for i in xrange(3)]
        im = cube[s].squeeze()
        l.set_data(im, **kwargs)
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()


# Too slow
def plot_3d(times):
    t = np.nonzero(times)

    # ax.scatter3D(times, c=zdata, cmap='Greens');
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(t[0], t[1], t[2], zdir='z', s=20, c=None, depthshade=True)

    plt.show()


def main(args):
    # Read the input file
    raw_spikes, cam_res, sim_time = read_recording_settings(args)
    times_debug = populate_debug_times(raw_spikes, cam_res, sim_time)
    cube_show_slider(times_debug)
    # plot_3d(times_debug)


if __name__ == '__main__':
    args = parse_args()
    main(args)