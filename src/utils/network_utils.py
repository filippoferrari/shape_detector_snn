# -*- coding: utf-8 -*-

from spikes_utils import neuron_id


def check_bounds(tmp, r1, r2):
    out = []
    for i in tmp:
        if i[0] >= 0 and i[0] < r1*r1 and i[1] >= 0 and i[1] < r2*r2:
            out.append(i)
    return out


def horizontal_connectivity_pos(r1, x, y, r2):
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
    tmp = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Positive
    tmp.append((neuron_id(x-1, y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y  , r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out


def horizontal_connectivity_neg(r1, x, y, r2):
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
    tmp = []
    out = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Negative
    tmp.append((neuron_id(x-2, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+2, y-1, r1), neuron_id(x2 , y2, r2)))
    
    tmp.append((neuron_id(x-2, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+2, y+1, r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out


def vertical_connectivity_pos(r1, x, y, r2):
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
    tmp = []
    out = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Positive
    tmp.append((neuron_id(x  , y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y+1, r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out


def vertical_connectivity_neg(r1, x, y, r2):
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
    tmp = []
    out = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Negative
    tmp.append((neuron_id(x-1, y-2, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y+2, r1), neuron_id(x2 , y2, r2)))
    
    tmp.append((neuron_id(x+1, y-2, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y+2, r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out


def left_diagonal_connectivity_pos(r1, x, y, r2):
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
    tmp = []
    out = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Positive
    tmp.append((neuron_id(x-1, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y+1, r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out


def left_diagonal_connectivity_neg(r1, x, y, r2):
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
    tmp = []
    out = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Negative
    tmp.append((neuron_id(x-2, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y+2, r1), neuron_id(x2 , y2, r2)))
    
    tmp.append((neuron_id(x-1, y-2, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+2, y+1, r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out


def right_diagonal_connectivity_pos(r1, x, y, r2):
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
    tmp = []
    out = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Positive
    tmp.append((neuron_id(x-1, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y-1, r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out


def right_diagonal_connectivity_neg(r1, x, y, r2):
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
    tmp = []
    out = []

    x2 = x / (r1 / r2)
    y2 = y / (r1 / r2)

    # Negative
    tmp.append((neuron_id(x-2, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x-1, y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y-1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y-2, r1), neuron_id(x2 , y2, r2)))
    
    tmp.append((neuron_id(x-1, y+2, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x  , y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y+1, r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+1, y  , r1), neuron_id(x2 , y2, r2)))
    tmp.append((neuron_id(x+2, y-1, r1), neuron_id(x2 , y2, r2)))

    out = check_bounds(tmp, r1, r2)

    return out
