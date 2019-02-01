# -*- coding: utf-8 -*-

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
    tmp = []

    for i in range(-stride, stride + 1):
        # Left side
        tmp.append((neuron_id(x-stride, y+i, r1), neuron_id(x, y, r2)))
        # Right side
        tmp.append((neuron_id(x+stride, y+i, r1), neuron_id(x, y, r2)))

    out = check_bounds(tmp, r1, r2)
    return out


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
    tmp = []

    for i in range(-stride, stride + 1):
        # Top side 
        tmp.append((neuron_id(x+i, y-stride, r1), neuron_id(x, y, r2)))
        # Bottom side
        tmp.append((neuron_id(x+i, y+stride, r1), neuron_id(x, y, r2)))

    out = check_bounds(tmp, r1, r2)
    return out


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
    |  ._________.
    |            stride
    |     (x,y)
    |   
    |  .         .
    +--------------------+

    """
    tmp = []

    for i in range(0, 2 * stride + 1):
        # Top side 
        tmp.append((neuron_id(x-2*stride+i, y+i, r1), neuron_id(x, y, r2)))
        # Bottom side
        tmp.append((neuron_id(x+i, y-2*stride+i, r1), neuron_id(x, y, r2)))

    out = check_bounds(tmp, r1, r2)
    return out


def right_diag_connections(r1, x, y, stride, r2):
    """
    Assume r1 is equal to r2

    Given an (x,y) point in a square of size r2 x r2 find the 
    neurons which model the right diagonal / sides of a square of size 
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
    tmp = []

    for i in range(0, 2 * stride + 1):
        # Top side 
        tmp.append((neuron_id(x-2*stride+i, y-i, r1), neuron_id(x, y, r2)))
        # Bottom side
        tmp.append((neuron_id(x+i, y+2*stride-i, r1), neuron_id(x, y, r2)))

    out = check_bounds(tmp, r1, r2)
    return out