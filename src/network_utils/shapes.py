# -*- coding: utf-8 -*-

from utils.constants import GAUSSIAN_WEIGHTS
from utils.spikes_utils import neuron_id, check_bounds


def vert_connections(r1, x, y, stride, r2):
    """
    Assume r1 is equal to r2

    Given an (x,y) point in a square of size r2 x r2 find the 
    neurons which model the vertical sides of a square of size 
    2 * stride + 1 in the square r1 x r1

               x
    +--------------------+
    |
    |
    |  .stride   .
    |  |         |
    |  |  (x,y)  |
    |  |         |
    |  .         .
    +--------------------+

    """
    out = []

    for j, w in enumerate(GAUSSIAN_WEIGHTS):
        gaussians = list(range(-(len(GAUSSIAN_WEIGHTS)//2), len(GAUSSIAN_WEIGHTS)//2 +1))
        for i in range(-stride, stride + 1):
            # Left side
            pre_x = x-stride+gaussians[j]
            pre_y = y+i
            out.append((neuron_id(pre_x, pre_y, r1), neuron_id(x, y, r2), w, 1))
            # Right side
            pre_x = x+stride+gaussians[j]
            pre_y = y+i
            out.append((neuron_id(pre_x, pre_y, r1), neuron_id(x, y, r2), w, 1))

    out = [i for i in out if i[0] != []]
    
    return list(set(out))


def hor_connections(r1, x, y, stride, r2):
    """
    Assume r1 is equal to r2

    Given an (x,y) point in a square of size r2 x r2 find the 
    neurons which model the horizontal sides of a square of size 
    2 * stride + 1 in the square r1 x r1

    In order to detect squares of different sizes, 5 sides are 
    computed, and the neurons on each of them have weights 
    determined by a gaussian distribution

               x
    +--------------------+
    |
    |
    |  ._________.
    |            stride
    |     (x,y)
    |   
    |  ._________.
    +--------------------+

    """
    out = []

    for j, w in enumerate(GAUSSIAN_WEIGHTS):
        gaussians = list(range(-(len(GAUSSIAN_WEIGHTS)//2), len(GAUSSIAN_WEIGHTS)//2 +1))
        for i in range(-stride, stride + 1):
            # Top side 
            pre_x = x+i
            pre_y = y-stride+gaussians[j]
            out.append((neuron_id(pre_x, pre_y, r1), neuron_id(x, y, r2), w, 1))
            # Bottom side
            pre_x = x+i
            pre_y = y+stride+gaussians[j]
            out.append((neuron_id(pre_x, pre_y, r1), neuron_id(x, y, r2), w, 1))

    out = [i for i in out if i[0] != []]

    return list(set(out))


def left_diag_connections(r1, x, y, stride, r2):
    """
    Assume r1 is equal to r2

    Given an (x,y) point in a square of size r2 x r2 find the 
    neurons which model the left diagonal \ sides of a square of size 
    2 * stride + 1 in the square r1 x r1

               x
    +--------------------+
    |     
    |
    |  .   (x,y)
    |   \
    |    \
    |     .
    |
    +--------------------+

    """
    out = []

    for i in range(0, 2 * stride + 1):
        # Left side 
        out.append((neuron_id(x-2*stride+i, y+i, r1), neuron_id(x, y, r2), 1, 1))
        # Right side
        out.append((neuron_id(x+i, y-2*stride+i, r1), neuron_id(x, y, r2), 1, 1))

    out = [i for i in out if i[0] != []]

    return out


def right_diag_connections(r1, x, y, stride, r2):
    """
    Assume r1 is equal to r2

    Given an (x,y) point in a square of size r2 x r2 find the 
    neurons which model the right diagonal / sides of a square of size 
    2 * stride + 1 in the square r1 x r1

               x
    +--------------------+
    |     . 
    |    / 
    |   /
    |  .   (x,y)
    |
    |
    |
    +--------------------+

    """
    out = []

    for i in range(0, 2 * stride + 1):
        # Top side 
        out.append((neuron_id(x-2*stride+i, y-i, r1), neuron_id(x, y, r2), 1, 1))
        # Bottom side
        out.append((neuron_id(x+i, y+2*stride-i, r1), neuron_id(x, y, r2), 1, 1))

    out = [i for i in out if i[0] != []]

    return out