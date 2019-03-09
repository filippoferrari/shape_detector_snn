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
    |  |
    |  |  (x,y)
    |  |
    |  .         .
    +--------------------+

    """
    out = []

    for j, w in enumerate(GAUSSIAN_WEIGHTS):
        gaussians = list(range(-(len(GAUSSIAN_WEIGHTS)//2), len(GAUSSIAN_WEIGHTS)//2 +1))
        #if x == 0 and y == 0:  print(' V -------------')
        for i in range(-stride, stride + 1):
            # Left side
            pre_x = x-stride+gaussians[j]
            pre_y = y+i
            #if x == 0 and y == 0: print(pre_x, pre_y)
            out.append((neuron_id(pre_x, pre_y, r1), neuron_id(x, y, r2), w, 1))
            # Right side
            pre_x = x+stride+gaussians[j]
            pre_y = y+i
            #if x == 0 and y == 0: print(pre_x, pre_y)
            out.append((neuron_id(pre_x, pre_y, r1), neuron_id(x, y, r2), w, 1))

    out = [i for i in out if i[0] != []]
    
    return list(set(out))


def hor_connections(r1, x, y, stride, r2):
    """
    Assume r1 is equal to r2

    Given an (x,y) point in a square of size r2 x r2 find the 
    neurons which model the horizontal sides of a square of size 
    2 * stride + 1 in the square r1 x r1

               x
    +--------------------+
    |
    |
    |  ._________.
    |            stride
    |     (x,y)
    |   
    |  .         .
    +--------------------+

    """
    out = []

    for j, w in enumerate(GAUSSIAN_WEIGHTS):
        gaussians = list(range(-(len(GAUSSIAN_WEIGHTS)//2), len(GAUSSIAN_WEIGHTS)//2 +1))
        #if x == 0 and y == 0:  print(' H -------------')
        for i in range(-stride, stride + 1):
            # Top side 
            pre_x = x+i
            pre_y = y-stride+gaussians[j]
            #if x == 0 and y == 0: print(pre_x, pre_y)
            out.append((neuron_id(pre_x, pre_y, r1), neuron_id(x, y, r2), w, 1))
            # Bottom side
            pre_x = x+i
            pre_y = y+stride+gaussians[j]
            #if x == 0 and y == 0: print(pre_x, pre_y)
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
        # Top side 
        out.append((neuron_id(x-2*stride+i, y+i, r1), neuron_id(x, y, r2)))
        # Bottom side
        out.append((neuron_id(x+i, y-2*stride+i, r1), neuron_id(x, y, r2)))

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
        out.append((neuron_id(x-2*stride+i, y-i, r1), neuron_id(x, y, r2)))
        # Bottom side
        out.append((neuron_id(x+i, y+2*stride-i, r1), neuron_id(x, y, r2)))

    out = [i for i in out if i[0] != []]

    return out