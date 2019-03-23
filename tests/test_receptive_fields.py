# -*- coding: utf-8 -*-

from src.network_utils.receptive_fields import *


def test_horizontal_positive():
    # Use a 12x12 camera for the tests
    cam_res = 12

    result = [(0, 0.0), (12, 0.0)]
    t = horizontal_connectivity_pos(cam_res, 0, 0, cam_res)
    assert result == t

    result = [(27, 39.0), (39, 39.0), (51, 39.0)]
    t = horizontal_connectivity_pos(cam_res, 3, 3, cam_res)
    assert result == t

    result = [(116, 128.0), (128, 128.0), (140, 128.0)]
    t = horizontal_connectivity_pos(cam_res, 10, 8, cam_res)
    assert result == t

    result = [(131, 143.0), (143, 143.0)]
    t = horizontal_connectivity_pos(cam_res, 11, 11, cam_res)
    assert result == t


def test_horizontal_negative():
    # Use a 12x12 camera for the tests
    cam_res = 12

    result = [(1, 0.0), (13, 0.0), (25, 0.0)]
    t = horizontal_connectivity_neg(cam_res, 0, 0, cam_res)
    assert result == t

    result = [(14, 39.0), (26, 39.0), (38, 39.0), (50, 39.0), (62, 39.0), \
              (16, 39.0), (28, 39.0), (40, 39.0), (52, 39.0), (64, 39.0)]
    t = horizontal_connectivity_neg(cam_res, 3, 3, cam_res)
    assert result == t

    result = [(103, 128.0), (115, 128.0), (127, 128.0), (139, 128.0),\
              (105, 128.0), (117, 128.0), (129, 128.0), (141, 128.0)]
    t = horizontal_connectivity_neg(cam_res, 10, 8, cam_res)
    assert result == t

    result = [(118, 143.0), (130, 143.0), (142, 143.0)]
    t = horizontal_connectivity_neg(cam_res, 11, 11, cam_res)
    assert result == t


def test_vertical_positive():
    # Use a 12x12 camera for the tests
    cam_res = 12

    result = [(0, 0.0), (1, 0.0)]
    t = vertical_connectivity_pos(cam_res, 0, 0, cam_res)
    assert result == t

    result = [(42, 43.0), (43, 43.0), (44, 43.0)]
    t = vertical_connectivity_pos(cam_res, 3, 7, cam_res)
    assert result == t

    result = [(120, 121.0), (121, 121.0), (122, 121.0)]
    t = vertical_connectivity_pos(cam_res, 10, 1, cam_res)
    assert result == t

    result = [(141, 142.0), (142, 142.0), (143, 142.0)]
    t = vertical_connectivity_pos(cam_res, 11, 10, cam_res)
    assert result == t


def test_vertical_negative():
    # Use a 12x12 camera for the tests
    cam_res = 12

    result = [(12, 0.0), (13, 0.0), (14, 0.0)]
    t = vertical_connectivity_neg(cam_res, 0, 0, cam_res)
    assert result == t

    result = [(29, 43.0), (30, 43.0), (31, 43.0),  (32, 43.0),  (33, 43.0),\
              (53, 43.0), (54, 43.0), (55, 43.0),  (56, 43.0),  (57, 43.0)]
    t = vertical_connectivity_neg(cam_res, 3, 7, cam_res)
    assert result == t

    result = [(108, 121.0), (109, 121.0), (110, 121.0), (111, 121.0),\
              (132, 121.0), (133, 121.0), (134, 121.0), (135, 121.0)]
    t = vertical_connectivity_neg(cam_res, 10, 1, cam_res)
    assert result == t

    result = [(128, 142.0), (129, 142.0), (130, 142.0), (131, 142.0)]
    t = vertical_connectivity_neg(cam_res, 11, 10, cam_res)
    assert result == t

