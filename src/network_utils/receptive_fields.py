# -*- coding: utf-8 -*-

from src.utils.spikes_utils import neuron_id, check_bounds


def filter_neurons(neurons):
    out = [i for i in neurons if i[0] != []]
    return out 
   

def horizontal_connectivity_pos(res1, x, y, res2):
    """
    Return the positive connections of the following 
    receptive field
    
                x
         -2 -1  0  1  2
      -2 
      -1  -  -  -  -  -
    y  0     +  +  + 
       1  -  -  -  -  -
       2

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Positive
    out.append((neuron_id(x-1, y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y  , res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out


def horizontal_connectivity_neg(res1, x, y, res2):
    """
    Return the negative connections of the following 
    receptive field

                x
         -2 -1  0  1  2
      -2 
      -1  -  -  -  -  -
    y  0     +  +  + 
       1  -  -  -  -  -
       2

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Negative
    out.append((neuron_id(x-2, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+2, y-1, res1), neuron_id(x2 , y2, res2)))
    
    out.append((neuron_id(x-2, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+2, y+1, res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out


def vertical_connectivity_pos(res1, x, y, res2):
    """
    Return the positive connections of the following 
    receptive field

                x
         -2 -1  0  1  2
      -2     -     -
      -1     -  +  -
    y  0     -  +  -
       1     -  +  - 
       2     -     -

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Positive
    out.append((neuron_id(x  , y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y+1, res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out


def vertical_connectivity_neg(res1, x, y, res2):
    """
    Return the negative connections of the following 
    receptive field

                x
         -2 -1  0  1  2
      -2     -     -
      -1     -  +  -
    y  0     -  +  -
       1     -  +  - 
       2     -     -

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Negative
    out.append((neuron_id(x-1, y-2, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y+2, res1), neuron_id(x2 , y2, res2)))
    
    out.append((neuron_id(x+1, y-2, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y+2, res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out


def left_diagonal_connectivity_pos(res1, x, y, res2):
    """
    Return the positive connections of the following 
    receptive field

                x
         -2 -1  0  1  2
      -2     - 
      -1  -  +  -  -
    y  0     -  +  -
       1     -  -  +  -
       2           -

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Positive
    out.append((neuron_id(x-1, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y+1, res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out


def left_diagonal_connectivity_neg(res1, x, y, res2):
    """
    Return the negative connections of the following 
    receptive field

                x
         -2 -1  0  1  2
      -2     - 
      -1  -  +  -  -
    y  0     -  +  -
       1     -  -  +  -
       2           -

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Negative
    out.append((neuron_id(x-2, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y+2, res1), neuron_id(x2 , y2, res2)))
    
    out.append((neuron_id(x-1, y-2, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+2, y+1, res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out


def right_diagonal_connectivity_pos(res1, x, y, res2):
    """
    Return the positive connections of the following 
    receptive field

                x
         -2 -1  0  1  2
      -2           -
      -1     -  -  +  -
    y  0     -  +  -
       1  -  +  -  -
       2     - 

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Positive
    out.append((neuron_id(x-1, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y-1, res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out


def right_diagonal_connectivity_neg(res1, x, y, res2):
    """
    Return the negative connections of the following 
    receptive field

                x
         -2 -1  0  1  2
      -2           -
      -1     -  -  +  -
    y  0     -  +  -
       1  -  +  -  -
       2     - 

    """
    out = []

    x2 = x / (res1 / res2)
    y2 = y / (res1 / res2)

    # Negative
    out.append((neuron_id(x-2, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x-1, y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y-1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y-2, res1), neuron_id(x2 , y2, res2)))
    
    out.append((neuron_id(x-1, y+2, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x  , y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y+1, res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+1, y  , res1), neuron_id(x2 , y2, res2)))
    out.append((neuron_id(x+2, y-1, res1), neuron_id(x2 , y2, res2)))

    out = filter_neurons(out)

    return out
