# -*- coding: utf-8 -*-

from src.network_utils.receptive_fields import *


def test_horizontal_positive():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(0, 0), (12, 0)]
    t = horizontal_connectivity_pos(cam_res, 0, 0, cam_res)
    assert sorted(result) == sorted(t)

    result = [(27, 39), (39, 39), (51, 39)]
    t = horizontal_connectivity_pos(cam_res, 3, 3, cam_res)
    assert sorted(result) == sorted(t)

    result = [(116, 128), (128, 128), (140, 128)]
    t = horizontal_connectivity_pos(cam_res, 10, 8, cam_res)
    assert sorted(result) == sorted(t)

    result = [(131, 143), (143, 143)]
    t = horizontal_connectivity_pos(cam_res, 11, 11, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = horizontal_connectivity_pos(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = horizontal_connectivity_pos(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = horizontal_connectivity_pos(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = horizontal_connectivity_pos(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)


def test_horizontal_negative():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(1, 0), (13, 0), (25, 0)]
    t = horizontal_connectivity_neg(cam_res, 0, 0, cam_res)
    assert sorted(result) == sorted(t)

    result = [(14, 39), (26, 39), (38, 39), (50, 39), (62, 39), \
              (16, 39), (28, 39), (40, 39), (52, 39), (64, 39)]
    t = horizontal_connectivity_neg(cam_res, 3, 3, cam_res)
    assert sorted(result) == sorted(t)

    result = [(103, 128), (115, 128), (127, 128), (139, 128),\
              (105, 128), (117, 128), (129, 128), (141, 128)]
    t = horizontal_connectivity_neg(cam_res, 10, 8, cam_res)
    assert sorted(result) == sorted(t)

    result = [(118, 143), (130, 143), (142, 143)]
    t = horizontal_connectivity_neg(cam_res, 11, 11, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = horizontal_connectivity_neg(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = horizontal_connectivity_neg(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = horizontal_connectivity_neg(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = horizontal_connectivity_neg(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)


def test_vertical_positive():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(0, 0), (1, 0)]
    t = vertical_connectivity_pos(cam_res, 0, 0, cam_res)
    assert sorted(result) == sorted(t)

    result = [(42, 43), (43, 43), (44, 43)]
    t = vertical_connectivity_pos(cam_res, 3, 7, cam_res)
    assert sorted(result) == sorted(t)

    result = [(120, 121), (121, 121), (122, 121)]
    t = vertical_connectivity_pos(cam_res, 10, 1, cam_res)
    assert sorted(result) == sorted(t)

    result = [(141, 142), (142, 142), (143, 142)]
    t = vertical_connectivity_pos(cam_res, 11, 10, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = vertical_connectivity_pos(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = vertical_connectivity_pos(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = vertical_connectivity_pos(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = vertical_connectivity_pos(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)


def test_vertical_negative():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(12, 0), (13, 0), (14, 0)]
    t = vertical_connectivity_neg(cam_res, 0, 0, cam_res)
    assert sorted(result) == sorted(t)

    result = [(29, 43), (30, 43), (31, 43),  (32, 43),  (33, 43),\
              (53, 43), (54, 43), (55, 43),  (56, 43),  (57, 43)]
    t = vertical_connectivity_neg(cam_res, 3, 7, cam_res)
    assert sorted(result) == sorted(t)

    result = [(108, 121), (109, 121), (110, 121), (111, 121),\
              (132, 121), (133, 121), (134, 121), (135, 121)]
    t = vertical_connectivity_neg(cam_res, 10, 1, cam_res)
    assert sorted(result) == sorted(t)

    result = [(128, 142), (129, 142), (130, 142), (131, 142)]
    t = vertical_connectivity_neg(cam_res, 11, 10, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = vertical_connectivity_neg(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = vertical_connectivity_neg(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = vertical_connectivity_neg(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = vertical_connectivity_neg(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)


def test_right_diagonal_positive():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(1, 12), (12, 12)]
    t = right_diagonal_connectivity_pos(cam_res, 1, 0, cam_res)
    assert sorted(result) == sorted(t)

    result = [(110, 121), (121, 121), (132, 121)]
    t = right_diagonal_connectivity_pos(cam_res, 10, 1, cam_res)
    assert sorted(result) == sorted(t)

    result = [(31, 42), (42, 42), (53, 42)]
    t = right_diagonal_connectivity_pos(cam_res, 3, 6, cam_res)
    assert sorted(result) == sorted(t)

    result = [(143, 143)]
    t = right_diagonal_connectivity_pos(cam_res, 11, 11, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = right_diagonal_connectivity_pos(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = right_diagonal_connectivity_pos(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = right_diagonal_connectivity_pos(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = right_diagonal_connectivity_pos(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)


def test_right_diagonal_negative():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(0, 12),\
              (2, 12), (13, 12), (25, 12), (24, 12)]
    t = right_diagonal_connectivity_neg(cam_res, 1, 0, cam_res)
    assert sorted(result) == sorted(t)

    result = [(98, 121), (109, 121), (108, 121),  (120, 121),\
              (111, 121), (122, 121), (134, 121),  (133, 121)]
    t = right_diagonal_connectivity_neg(cam_res, 10, 1, cam_res)
    assert sorted(result) == sorted(t)

    result = [(19, 42), (30, 42), (29, 42), (41, 42), (52, 42),\
              (32, 42), (43, 42), (55, 42), (54, 42), (65, 42)]
    t = right_diagonal_connectivity_neg(cam_res, 3, 6, cam_res)
    assert sorted(result) == sorted(t)

    result = [(131, 143), (130, 143), (142, 143)]
    t = right_diagonal_connectivity_neg(cam_res, 11, 11, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = right_diagonal_connectivity_neg(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = right_diagonal_connectivity_neg(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = right_diagonal_connectivity_neg(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = right_diagonal_connectivity_neg(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)


def test_left_diagonal_positive():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(0, 13), (13, 13), (26, 13)]
    t = left_diagonal_connectivity_pos(cam_res, 1, 1, cam_res)
    assert sorted(result) == sorted(t)

    result = [(86, 99), (99, 99), (112, 99)]
    t = left_diagonal_connectivity_pos(cam_res, 8, 3, cam_res)
    assert sorted(result) == sorted(t)

    result = [(34, 47), (47, 47)]
    t = left_diagonal_connectivity_pos(cam_res, 3, 11, cam_res)
    assert sorted(result) == sorted(t)

    result = [(130, 143), (143, 143)]
    t = left_diagonal_connectivity_pos(cam_res, 11, 11, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = left_diagonal_connectivity_pos(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = left_diagonal_connectivity_pos(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = left_diagonal_connectivity_pos(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = left_diagonal_connectivity_pos(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)


def test_left_diagonal_negative():
    # Use a 12x12 camera for the tests
    cam_res = 12

    #### Pixels inside the frame

    result = [(12, 13), (24, 13), (25, 13), (38, 13),\
              (1, 13), (2, 13), (14, 13), (27, 13)]
    t = left_diagonal_connectivity_neg(cam_res, 1, 1, cam_res)
    assert sorted(result) == sorted(t)

    result = [(85, 99), (98, 99), (110, 99),  (111, 99),  (124, 99),\
              (74, 99), (87, 99), (88, 99),  (100, 99),  (113, 99)]
    t = left_diagonal_connectivity_neg(cam_res, 8, 3, cam_res)
    assert sorted(result) == sorted(t)

    result = [(33, 47), (46, 47), (58, 47), (59, 47),\
              (22, 47), (35, 47)]
    t = left_diagonal_connectivity_neg(cam_res, 3, 11, cam_res)
    assert sorted(result) == sorted(t)

    result = [(129, 143), (142, 143), (118, 143), (131, 143)]
    t = left_diagonal_connectivity_neg(cam_res, 11, 11, cam_res)
    assert sorted(result) == sorted(t)

    #### Pixels outside 

    result = []
    t = left_diagonal_connectivity_neg(cam_res, -1, -1, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = left_diagonal_connectivity_neg(cam_res, -1, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = left_diagonal_connectivity_neg(cam_res, 13, 4, cam_res)
    assert sorted(result) == sorted(t)

    result = []
    t = left_diagonal_connectivity_neg(cam_res, 13, 14, cam_res)
    assert sorted(result) == sorted(t)