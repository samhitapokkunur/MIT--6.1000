from functools import wraps
import io
import os
import pset
import random
import re
import string
import sys
import unittest
from unittest.mock import MagicMock, patch


############################################################
# set up test case examples
############################################################


## ----- 2.1 test dict ---- ##

solar_panel_filename = "solar_panel_data.txt"
solar_panel_x = [0.0, 1.8, 4.7, 6.2, 7.5, 9.9, 11.6, 12.8, 15.4, 17.9, 21.0, 22.9, 25.6, 28.2, 30.4, 33.7, 35.9, 38.8, 42.0, 44.7, 47.5, 50.5, 53.7, 56.9, 60.2, 63.4, 66.1, 70.1, 72.8, 75.9, 79.7, 82.5, 86.6, 90.8, 94.9, 99.5]
solar_panel_y = [100.0, 97.6, 96.2, 94.9, 99.1, 93.7, 91.2, 88.0, 95.4, 90.1, 87.6, 83.5, 85.1, 80.4, 86.9, 78.0, 76.5, 81.7, 71.2, 77.8, 72.4, 66.9, 74.6, 68.5, 71.9, 65.7, 62.9, 68.3, 60.2, 63.8, 55.5, 61.1, 57.4, 64.9, 52.8, 59.0]
solar_panel_ans = [solar_panel_x, solar_panel_y]
solar_panel_dict = {solar_panel_filename: solar_panel_ans}


## ----- 2.2 test dict ---- ##

mse_cases = {
    "perfect_prediction": {
        "y_actual": [1.0, 2.0, 3.0],
        "y_pred": [1.0, 2.0, 3.0],
        "expected_mse": 0.0
    },
    "small_errors": {
        "y_actual": [1.0, 2.0, 3.0],
        "y_pred": [1.1, 1.9, 3.2],  # diffs: -0.1, 0.1, -0.2
        "expected_mse": (0.01 + 0.01 + 0.04) / 3  # = 0.02
    },
    "negative_values": {
        "y_actual": [-1.0, -2.0, -3.0],
        "y_pred": [-1.0, -1.0, -2.0],  # diffs: 0, -1, -1
        "expected_mse": (0.0 + 1.0 + 1.0) / 3  # = 0.666...
    },
    "large_values": {
        "y_actual": [100.0, 200.0, 300.0],
        "y_pred": [110.0, 190.0, 290.0],  # diffs: -10, 10, 10
        "expected_mse": (100.0 + 100.0 + 100.0) / 3  # = 100.0
    },
    "mixed_signs": {
        "y_actual": [2.0, -2.0, 2.0, -2.0],
        "y_pred": [-2.0, 2.0, -2.0, 2.0],  # diffs: 4, -4, 4, -4
        "expected_mse": (16.0 + 16.0 + 16.0 + 16.0) / 4  # = 16.0
    },
    "single_element": {
        "y_actual": [5.0],
        "y_pred": [7.0],  # diff = -2
        "expected_mse": 4.0
    },
    "all_wrong": {
        "y_actual": [0.0, 0.0, 0.0],
        "y_pred": [1.0, 2.0, 3.0],  # diffs: -1, -2, -3
        "expected_mse": (1.0 + 4.0 + 9.0) / 3  # = 14/3 ≈ 4.666...
    }
}


## ----- 2.3 test dict ---- ##

fit_linear_small = {
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
    "a_range": [0, 3],
    "b_range": [0, 2],
    "precision": 0.005,
    "expected": (2.0, 1.0),
}

fit_linear_big1 = {
    "x": [-1.25, 4.51, 2.32, 0.99, -3.44, -3.44, -4.42, 3.66, 1.01, 2.08, -4.79, 4.7, 3.32, -2.88, -3.18, -3.17, -1.96, 0.25, -0.68, -2.09, 1.12, -3.61, -2.08, -1.34, -0.44, 2.85, -3.0, 0.14, 0.92, -4.54, 1.08, -3.29, -4.35, 4.49, 4.66, 3.08, -1.95, -4.02, 1.84, -0.6, -3.78, -0.05, -4.66, 4.09, -2.41, 1.63, -1.88, 0.2, 0.47, -3.15, 4.7, 2.75, 4.39, 3.95, 0.98, 4.22, -4.12, -3.04, -4.55, -1.75, -1.11, -2.29, 3.29, -1.43, -2.19, 0.43, -3.59, 3.02, -4.25, 4.87, 2.72, -3.01, -4.94, 3.15, 2.07, 2.29, 2.71, -4.26, -1.42, -3.84, 3.63, 1.23, -1.69, -4.36, -1.89, -1.75, 2.3, 1.38, 3.87, -0.28, -3.8, 2.13, 2.61, 0.61, 2.71, -0.06, 0.23, -0.72, -4.75, -3.92],
    "y": [-1.6, 12.7, 7.32, 3.48, -7.15, -7.01, -9.18, 10.52, 3.82, 6.57, -10.25, 13.33, 9.67, -5.57, -6.43, -6.18, -3.58, 2.04, -0.3, -4.09, 4.37, -7.46, -3.7, -1.91, 0.05, 8.52, -6.09, 1.65, 3.76, -9.75, 4.67, -6.68, -9.31, 12.71, 12.67, 9.19, -3.36, -7.93, 6.05, 0.08, -7.96, 1.08, -9.86, 11.91, -4.33, 5.35, -2.85, 1.65, 2.82, -5.83, 13.0, 8.23, 12.5, 11.25, 3.56, 12.07, -9.07, -5.98, -10.1, -2.49, -1.47, -4.31, 9.93, -2.38, -3.92, 2.9, -7.88, 9.1, -9.06, 13.87, 7.99, -6.36, -10.72, 9.45, 6.74, 7.31, 8.1, -9.09, -1.98, -8.28, 11.04, 4.69, -3.02, -9.24, -3.47, -2.68, 7.54, 4.74, 11.42, 0.9, -7.79, 7.3, 7.96, 2.84, 8.05, 1.15, 2.06, -0.21, -10.31, -8.09],
    "a_range": [2.4, 2.6],
    "b_range": [1.4, 1.6],
    "precision": 0.005,
    "expected": (2.490, 1.495),
}

fit_linear_big2 = {
    "x": [-1.25, 4.51, 2.32, 0.99, -3.44, -3.44, -4.42, 3.66, 1.01, 2.08, -4.79, 4.7, 3.32, -2.88, -3.18, -3.17, -1.96, 0.25, -0.68, -2.09, 1.12, -3.61, -2.08, -1.34, -0.44, 2.85, -3.0, 0.14, 0.92, -4.54, 1.08, -3.29, -4.35, 4.49, 4.66, 3.08, -1.95, -4.02, 1.84, -0.6, -3.78, -0.05, -4.66, 4.09, -2.41, 1.63, -1.88, 0.2, 0.47, -3.15, 4.7, 2.75, 4.39, 3.95, 0.98, 4.22, -4.12, -3.04, -4.55, -1.75, -1.11, -2.29, 3.29, -1.43, -2.19, 0.43, -3.59, 3.02, -4.25, 4.87, 2.72, -3.01, -4.94, 3.15, 2.07, 2.29, 2.71, -4.26, -1.42, -3.84, 3.63, 1.23, -1.69, -4.36, -1.89, -1.75, 2.3, 1.38, 3.87, -0.28, -3.8, 2.13, 2.61, 0.61, 2.71, -0.06, 0.23, -0.72, -4.75, -3.92],
    "y": [-1.6, 12.7, 7.32, 3.48, -7.15, -7.01, -9.18, 10.52, 3.82, 6.57, -10.25, 13.33, 9.67, -5.57, -6.43, -6.18, -3.58, 2.04, -0.3, -4.09, 4.37, -7.46, -3.7, -1.91, 0.05, 8.52, -6.09, 1.65, 3.76, -9.75, 4.67, -6.68, -9.31, 12.71, 12.67, 9.19, -3.36, -7.93, 6.05, 0.08, -7.96, 1.08, -9.86, 11.91, -4.33, 5.35, -2.85, 1.65, 2.82, -5.83, 13.0, 8.23, 12.5, 11.25, 3.56, 12.07, -9.07, -5.98, -10.1, -2.49, -1.47, -4.31, 9.93, -2.38, -3.92, 2.9, -7.88, 9.1, -9.06, 13.87, 7.99, -6.36, -10.72, 9.45, 6.74, 7.31, 8.1, -9.09, -1.98, -8.28, 11.04, 4.69, -3.02, -9.24, -3.47, -2.68, 7.54, 4.74, 11.42, 0.9, -7.79, 7.3, 7.96, 2.84, 8.05, 1.15, 2.06, -0.21, -10.31, -8.09],
    "a_range": [2.2, 2.6],
    "b_range": [1.2, 1.6],
    "precision": 0.005,
    "expected": (2.490, 1.495),
}

fit_linear_solar = {
    "x": solar_panel_x,
    "y": solar_panel_y,
    "a_range": [-5, 0],
    "b_range": [0, 100],
    "precision": 0.1,
    "expected": (-0.5, 98.9),
}

fit_linear_cases = {
    "small": fit_linear_small,
    "big1": fit_linear_big1,
    "big2": fit_linear_big2,
    "solar": fit_linear_solar,
}


## ----- 3.1 test dict ---- ##

input_x = ['0.47', '0.5%', '#4%.5', '0.94', '1.41', '1.88', '2.35', '$$', '2.82', '3.29', '3.76', '4.23', '4.70', '5.17', '&.34', '5.64', '6.11', '6.58', '7.05', '7.30', 'ERROR', '#$%', '7.52', '7.99', '8.46', 'ERROR', '####', '8.93', '9.40', '9.87', '@%$%', '10.21', '10.34', '10.81', '11.28', '####']
input_y = ['12.8', '15.3', '####', '42.5', '25.9', '78.3', '65.7', '75.6', '131.4', '117.2', '150.5', '141.7', '180.2', '155.6', ' 167.5', '192.1', '174.5', '161.8', '187.3', '@$%%@', '170.4', '####', '160.5', '178.2', '143.8', 'ERROR_RIP', '#$#', '163.4', '132.7', '120.3', ' ERROR', ' 80.$', '91.2', '70.5', '39.8', '####']
input_plant_data = [input_x, input_y]
x_vals_ans = ('0.47', '0.94', '1.41', '1.88', '2.35', '2.82', '3.29', '3.76', '4.23', '4.70', '5.17', '5.64', '6.11', '6.58', '7.05', '7.52', '7.99', '8.46', '8.93', '9.40', '9.87', '10.34', '10.81', '11.28')
y_vals_ans = ('12.8', '42.5', '25.9', '78.3', '65.7', '131.4', '117.2', '150.5', '141.7', '180.2', '155.6', '192.1', '174.5', '161.8', '187.3', '160.5', '178.2', '143.8', '163.4', '132.7', '120.3', '91.2', '70.5', '39.8')
plant_data_ans = (x_vals_ans, y_vals_ans)

plant_test_x_ans = ('6.22', '6.2', '7.29', '5.97', '1.45', '9.56', '2.22', '0.87', '3.56', '4.96', '3.36', '5.98', '1.34')
plant_test_y_ans = ('7.61', '2.32', '1.8', '1.23', '1.42', '5.37', '2.15', '4.02', '7.29', '5.36', '3.25', '7.21', '5.92')
plant_test_ans = (plant_test_x_ans, plant_test_y_ans)
input_test_x = ['6.22', '6.20', '7.29', '5.97', 'ik.', '.g4', '1.45', '9.56', '5.97', '2.22', '0.87', '3.56', '4.96', '3.36', '5.80', '5.98', '.dt', '?.y', '.Sc', '1.34']
input_test_y = ['7.61', '2.32', '1.80', '1.23', '7.57', '0.22', '1.42', '5.37', '#k3', '2.15', '4.02', '7.29', '5.36', '3.25', 'dW0', '7.21', '0.22', '0.05', '8.81', '5.92']
input_plant_test = [input_test_x, input_test_y]

alr_clean_x = ['6.22', '6.2', '7.29', '5.97', '1.45', '9.56', '2.22', '0.87', '3.56', '4.96', '3.36', '5.98', '1.34']
alr_clean_y = ['7.61', '2.32', '1.8', '1.23', '1.42', '5.37', '2.15', '4.02', '7.29', '5.36', '3.25', '7.21', '5.92']
alr_clean_test = [alr_clean_x, alr_clean_y]

all_malformed_x = ['8db', 'oZg', 'ict', '5$b', '4bd', 'z.#', 'erz', 'l.0', '.d1', 'rme', '.e%', 'gc.', 'bg4', '$j#', 'ng?', '$.h', 'hWl', 'fYy', 'slp', 'y$a']
all_malformed_y = ['lqa', 'rq.', 'eUm', '2Wb', '1&z', 'a.k', '*ft', '?q6', '7.2', 'm@h', '?ot', '#g&', 'bWs', 'kn$', 'c..', '*kr', '1Zp', 'r.z', '4Ye', '&ep']
all_malformed_test = (all_malformed_x, all_malformed_y)
all_malformed_ans = ((),())

clean_data_dict = {plant_data_ans: input_plant_data,
                    plant_test_ans: input_plant_test,
                    all_malformed_ans: all_malformed_test,
                    plant_test_ans: alr_clean_test}


## ----- 3.2 test dict ---- ##

fit_quadratic_cases = {
    "parabola": {
        "x": [-2, -1, 0, 1, 2],
        "y": [4, 1, 0, 1, 4],                # y = x^2
        "a_range": [0, 2],
        "b_range": [-1, 1],
        "c_range": [-1, 1],
        "precision": 0.1,
        "expected": (1, 0, 0),
    },
    "fractional": {
        "x": [0.0, 0.5, 1.0, 1.5],
        "y": [2.0, 1.125, 1.0, 1.625],       # y = 1.5x^2 - 2.5x + 2
        "a_range": [0, 5],
        "b_range": [-5, 0],
        "c_range": [0, 3],
        "precision": 0.05,
        "expected": (1.5, -2.5, 2),
    },
    # negative a (opens downward)
    "downward": {
        "x": [-1, 0, 1, 2],
        "y": [0, 1, 0, -3],                  # y = -x^2 + 1
        "a_range": [-2, 0],
        "b_range": [-5, 5],
        "c_range": [0, 2],
        "precision": 0.1,
        "expected": (-1, 0, 1),
    },
    # small noisy dataset (approximate best fit)
    "small_noisy": {
        "x": [0, 1, 2, 3],
        "y": [1.1, 1.9, 6.2, 11.1],          # noisy y = x^2 + 1
        "a_range": [0, 2],
        "b_range": [0, 2],
        "c_range": [0, 2],
        "precision": 0.1,
        "expected": (1, 0.4, 1),
    },
    # your larger real-worldish case
    "plant": {
        "x": [
            0.47, 0.94, 1.41, 1.88, 2.35, 2.82, 3.29, 3.76, 4.23, 4.7,
            5.17, 5.64, 6.11, 6.58, 7.05, 7.52, 7.99, 8.46, 8.93, 9.4,
            9.87, 10.34, 10.81, 11.28
        ],
        "y": [
            12.8, 42.5, 25.9, 78.3, 65.7, 131.4, 117.2, 150.5, 141.7, 180.2,
            155.6, 192.1, 174.5, 161.8, 187.3, 160.5, 178.2, 143.8, 163.4, 132.7,
            120.3, 91.2, 70.5, 39.8
        ],
        "a_range": [-7.0, -5.0],
        "b_range": [60.0, 70.0],
        "c_range": [-35.0, -30.0],
        "precision": 0.1,
        "expected": (-5.2, 66.0, -31.6),
    },
}

predict_y_cases = {
    "linear": {

        "zero": {
            "x-vals": [0.0, 1.3, 3.5, 6.6, 23.4, 87.6],
            "coefficients": [0.0, 0.0],       # y = 0
            "expected_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "slope_only": {
            "x-vals": [0.0, 1.3, 3.5, 6.6, 23.4, 87.6],
            "coefficients": [1.0, 0.0],       # y = 1.0 * x
            "expected_y": [0.0, 1.3, 3.5, 6.6, 23.4, 87.6]
        },
        "intercept_only": {
            "x-vals": [-5.0, 0.0, 5.0],
            "coefficients": [0.0, -1.0],      # y = -1
            "expected_y": [-1.0, -1.0, -1.0]
        },
        "negative_slope": {
            "x-vals": [-3.2, -1.4, 1.2, 2.6, 4.1, 5.7, 7.3, 9.0],
            "coefficients": [-3.0, 4.0],      # y = -3x + 4
            "expected_y": [13.6, 8.2, 0.4, -3.8, -8.3, -13.1, -17.9, -23.0]
        },
        "large_values": {
            "x-vals": [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
            "coefficients": [-1.5, -20.0],    # y = -1.5x - 20
            "expected_y": [10.0, 2.5, -5.0, -12.5, -20.0, -27.5, -35.0, -42.5, -50.0]
        }
    },
    "quadratic": {

        "zero": {
            "x-vals": [0.0, 1.3, 3.5, 6.6, 23.4, 87.6],
            "coefficients": [0.0, 0.0, 0.0],
            "expected_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "pure_quadratic": {
            "x-vals": [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0],
            "coefficients": [2.0, 0.0, 0.0],  # y = 2x**2
            "expected_y": [18.0, 2.0, 0.0, 2.0, 8.0, 50.0]
        },
        "pure_linear": {
            "x-vals": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "coefficients": [0.0, -3.0, 0.0],  # y = -3x
            "expected_y": [6.0, 3.0, 0.0, -3.0, -6.0]
        },
        "pure_constant": {
            "x-vals": [0.0, 5.0, 10.0],
            "coefficients": [0.0, 0.0, 7.5],  # y = 7.5
            "expected_y": [7.5, 7.5, 7.5]
        },
        "mixed_positive_negative": {
            "x-vals": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "coefficients": [1.0, -2.0, 1.0],  # y = x**2 - 2x + 1
            "expected_y": [9.0, 4.0, 1.0, 0.0, 1.0]
        },
        "non_integer_coefficients": {
            "x-vals": [0.5, 1.5, 2.5],
            "coefficients": [0.5, -1.25, 3.0],  # y = 0.5x**2 -1.25x + 3
            "expected_y": [2.5, 2.25, 3.0]
        },
        "large_x_values": {
            "x-vals": [100.0, 250.0, 1000.0],
            "coefficients": [0.1, 0.0, 0.0],  # y = 0.1x**2
            "expected_y": [1000.0, 6250.0, 100000.0]
        },
    },
}


## ----- 4 test dict ---- ##

fit_linear_bisect_small = {
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
    "a_range": [0, 3],
    "precision": 5e-6,
    "expected": (1.99951171875, 1.001220703125),
}

fit_linear_bisect_big1 = {
    "x": [-1.25, 4.51, 2.32, 0.99, -3.44, -3.44, -4.42, 3.66, 1.01, 2.08, -4.79, 4.7, 3.32, -2.88, -3.18, -3.17, -1.96, 0.25, -0.68, -2.09, 1.12, -3.61, -2.08, -1.34, -0.44, 2.85, -3.0, 0.14, 0.92, -4.54, 1.08, -3.29, -4.35, 4.49, 4.66, 3.08, -1.95, -4.02, 1.84, -0.6, -3.78, -0.05, -4.66, 4.09, -2.41, 1.63, -1.88, 0.2, 0.47, -3.15, 4.7, 2.75, 4.39, 3.95, 0.98, 4.22, -4.12, -3.04, -4.55, -1.75, -1.11, -2.29, 3.29, -1.43, -2.19, 0.43, -3.59, 3.02, -4.25, 4.87, 2.72, -3.01, -4.94, 3.15, 2.07, 2.29, 2.71, -4.26, -1.42, -3.84, 3.63, 1.23, -1.69, -4.36, -1.89, -1.75, 2.3, 1.38, 3.87, -0.28, -3.8, 2.13, 2.61, 0.61, 2.71, -0.06, 0.23, -0.72, -4.75, -3.92],
    "y": [-1.6, 12.7, 7.32, 3.48, -7.15, -7.01, -9.18, 10.52, 3.82, 6.57, -10.25, 13.33, 9.67, -5.57, -6.43, -6.18, -3.58, 2.04, -0.3, -4.09, 4.37, -7.46, -3.7, -1.91, 0.05, 8.52, -6.09, 1.65, 3.76, -9.75, 4.67, -6.68, -9.31, 12.71, 12.67, 9.19, -3.36, -7.93, 6.05, 0.08, -7.96, 1.08, -9.86, 11.91, -4.33, 5.35, -2.85, 1.65, 2.82, -5.83, 13.0, 8.23, 12.5, 11.25, 3.56, 12.07, -9.07, -5.98, -10.1, -2.49, -1.47, -4.31, 9.93, -2.38, -3.92, 2.9, -7.88, 9.1, -9.06, 13.87, 7.99, -6.36, -10.72, 9.45, 6.74, 7.31, 8.1, -9.09, -1.98, -8.28, 11.04, 4.69, -3.02, -9.24, -3.47, -2.68, 7.54, 4.74, 11.42, 0.9, -7.79, 7.3, 7.96, 2.84, 8.05, 1.15, 2.06, -0.21, -10.31, -8.09],
    "a_range": [2.4, 2.6],
    "precision": 5e-6,
    "expected": (2.4890624999999997, 1.4960895312499993),
}

fit_linear_bisect_big2 = {
    "x": [-1.25, 4.51, 2.32, 0.99, -3.44, -3.44, -4.42, 3.66, 1.01, 2.08, -4.79, 4.7, 3.32, -2.88, -3.18, -3.17, -1.96, 0.25, -0.68, -2.09, 1.12, -3.61, -2.08, -1.34, -0.44, 2.85, -3.0, 0.14, 0.92, -4.54, 1.08, -3.29, -4.35, 4.49, 4.66, 3.08, -1.95, -4.02, 1.84, -0.6, -3.78, -0.05, -4.66, 4.09, -2.41, 1.63, -1.88, 0.2, 0.47, -3.15, 4.7, 2.75, 4.39, 3.95, 0.98, 4.22, -4.12, -3.04, -4.55, -1.75, -1.11, -2.29, 3.29, -1.43, -2.19, 0.43, -3.59, 3.02, -4.25, 4.87, 2.72, -3.01, -4.94, 3.15, 2.07, 2.29, 2.71, -4.26, -1.42, -3.84, 3.63, 1.23, -1.69, -4.36, -1.89, -1.75, 2.3, 1.38, 3.87, -0.28, -3.8, 2.13, 2.61, 0.61, 2.71, -0.06, 0.23, -0.72, -4.75, -3.92],
    "y": [-1.6, 12.7, 7.32, 3.48, -7.15, -7.01, -9.18, 10.52, 3.82, 6.57, -10.25, 13.33, 9.67, -5.57, -6.43, -6.18, -3.58, 2.04, -0.3, -4.09, 4.37, -7.46, -3.7, -1.91, 0.05, 8.52, -6.09, 1.65, 3.76, -9.75, 4.67, -6.68, -9.31, 12.71, 12.67, 9.19, -3.36, -7.93, 6.05, 0.08, -7.96, 1.08, -9.86, 11.91, -4.33, 5.35, -2.85, 1.65, 2.82, -5.83, 13.0, 8.23, 12.5, 11.25, 3.56, 12.07, -9.07, -5.98, -10.1, -2.49, -1.47, -4.31, 9.93, -2.38, -3.92, 2.9, -7.88, 9.1, -9.06, 13.87, 7.99, -6.36, -10.72, 9.45, 6.74, 7.31, 8.1, -9.09, -1.98, -8.28, 11.04, 4.69, -3.02, -9.24, -3.47, -2.68, 7.54, 4.74, 11.42, 0.9, -7.79, 7.3, 7.96, 2.84, 8.05, 1.15, 2.06, -0.21, -10.31, -8.09],
    "a_range": [2.2, 2.6],
    "precision": 5e-6,
    "expected": (2.4890624999999997, 1.4960895312499993),
}

fit_linear_bisect_solar = {
    "x": solar_panel_x,
    "y": solar_panel_y,
    "a_range": [-1, 0],
    "precision": 5e-6,
    "expected": (-0.451171875, 96.89892749256555),
}

fit_linear_bisect_cases = {
    "small": fit_linear_bisect_small,
    "big1": fit_linear_bisect_big1,
    "big2": fit_linear_bisect_big2,
    "solar": fit_linear_bisect_solar,
}


############################################################
# test case settings
############################################################


# DO NOT MODIFY
def case_options(points, failure, error):
    """Decorator to add points and messages to a test case"""

    def decorator(func):
        # Directly set attributes on the original function
        func.points = points
        func.failure_message = failure
        func.error_message = error

        @wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(args[-1], MagicMock):
                args = args[:-1]
            return func(*args, **kwargs)

        return wrapper

    return decorator


# DO NOT MODIFY
def testsuite_options(timeout, weight):
    """Decorator to add timeout and weight to a test suite"""
    # print("timeout", timeout, "weight", weight)

    def decorator(cls):
        # Directly set attributes on the original class
        cls.timeout = timeout
        cls.weight = weight
        return cls

    return decorator


############################################################
# part 2.1: reading data
############################################################


@testsuite_options(4, 1)
class TestPart2_1(unittest.TestCase):
    """Test read data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        2,
        "Your code does not read the data correctly",
        "Task test_read_data error",
    )
    def test_read_data(self):
        for filename, read_data_done in solar_panel_dict.items():
            actual = pset.read_data(filename)
            self.assertEqual(
                read_data_done,
                actual,
                f"Incorrect cleaned data, Expected: {read_data_done}, got: {actual}",
            )


############################################################
# part 2.2: getting mean squared error
############################################################


@testsuite_options(4, 1)
class TestPart2_2(unittest.TestCase):
    """Test mean-squared error"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        3,
        "Your test_mean_squared_error function does not find the correct MSE",
        "Task test_mean_squared_error error",
    )
    def test_mean_squared_error(self):
        for case_name, case in mse_cases.items():
            calculated_mse = pset.mean_squared_error(case["y_actual"], case["y_pred"])
            self.assertAlmostEqual(calculated_mse, case["expected_mse"], places=3,
                msg=f"Incorrect MSE for {case_name} case. Expected: {case['expected_mse']}, got: {calculated_mse}")


############################################################
# part 2.3: fitting a linear model
############################################################


@testsuite_options(10, 1)
class TestPart2_3(unittest.TestCase):
    """Test fit linear model and predict y"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        1,
        "Your fit_linear_model function does not find the correct coefficients for small linear data",
        "Task test_fit_linear_model_edge_case_1 error",
    )
    def test_fit_linear_model_small_1(self):
        case = fit_linear_cases["small"]
        a_optimal, b_optimal = pset.fit_linear_model(
            case["x"], case["y"], case["a_range"], case["b_range"], case["precision"]
        )

        self.assertAlmostEqual(a_optimal, case["expected"][0], delta=case["precision"] * 2,
                         msg=f"Expected a ≈ {case['expected'][0]}, got {a_optimal}")

        self.assertAlmostEqual(b_optimal, case["expected"][1], delta=case["precision"] * 2,
                            msg=f"Expected b ≈ {case['expected'][1]}, got {b_optimal}")

    @case_options(
        1,
        "Your fit_linear_model function does not find the correct coefficients for big linear data 1",
        "Task test_fit_linear_model_big_1 error",
    )
    def test_fit_linear_model_big_1(self):
        case = fit_linear_cases["big1"]
        a_optimal, b_optimal = pset.fit_linear_model(
            case["x"], case["y"], case["a_range"], case["b_range"], case["precision"]
        )

        self.assertAlmostEqual(a_optimal, case["expected"][0], delta=case["precision"] * 2,
                         msg=f"Expected a ≈ {case['expected'][0]}, got {a_optimal}")

        self.assertAlmostEqual(b_optimal, case["expected"][1], delta=case["precision"] * 2,
                            msg=f"Expected b ≈ {case['expected'][1]}, got {b_optimal}")

    @case_options(
        1,
        "Your fit_linear_model function does not find the correct coefficients for big linear data 2",
        "Task test_fit_linear_model_big_2 error",
    )
    def test_fit_linear_model_big_2(self):
        case = fit_linear_cases["big2"]
        a_optimal, b_optimal = pset.fit_linear_model(
            case["x"], case["y"], case["a_range"], case["b_range"], case["precision"]
        )

        self.assertAlmostEqual(a_optimal, case["expected"][0], delta=case["precision"] * 2,
                         msg=f"Expected a ≈ {case['expected'][0]}, got {a_optimal}")

        self.assertAlmostEqual(b_optimal, case["expected"][1], delta=case["precision"] * 2,
                            msg=f"Expected b ≈ {case['expected'][1]}, got {b_optimal}")

    @case_options(
        1,
        "Your fit_linear_model function does not find the correct coefficients for the solar panel data",
        "Task test_fit_linear_model_solar error",
    )
    def test_fit_linear_model_solar(self):
        case = fit_linear_cases["solar"]
        a_optimal, b_optimal = pset.fit_linear_model(
            case["x"], case["y"], case["a_range"], case["b_range"], case["precision"]
        )

        self.assertAlmostEqual(a_optimal, case["expected"][0], delta=case["precision"] * 2,
                         msg=f"Expected a ≈ {case['expected'][0]}, got {a_optimal}")

        self.assertAlmostEqual(b_optimal, case["expected"][1], delta=case["precision"] * 2,
                            msg=f"Expected b ≈ {case['expected'][1]}, got {b_optimal}")

    @case_options(
        1,
        "Your predict_y function does not find the correct y-values for linear functions",
        "Task test_predict_y_linear error",
    )
    def test_predict_y_linear(self):
        linear_cases = predict_y_cases["linear"]
        for case_name, case in linear_cases.items():
            predicted_y = []
            for x in case["x-vals"]:
                predicted_y.append(pset.predict_y(x, case["coefficients"]))

            # check each value in predicted_y against expected_y
            for pred, exp in zip(predicted_y, case["expected_y"]):
                self.assertAlmostEqual(pred, exp, places=3,
                    msg=f"Incorrect predicted y values for {case_name} linear case. Expected: {case['expected_y']}, got: {predicted_y}")


############################################################
# part 3.1: data cleaning
############################################################


@testsuite_options(4, 1)
class TestPart3_1(unittest.TestCase):
    """Test clean data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @case_options(
        3,
        "Your code does not clean data correctly",
        "Task test_clean_data error",
    )
    def test_clean_data(self):
        for clean_data_done, input_plant_data in clean_data_dict.items():
            [x_val, y_val] = input_plant_data # x_val, y_val are lists
            starting_id = [id(x_val), id(y_val)]
            pset.clean_data(x_val, y_val) # call student's clean_data on x and y
            new_id = [id(x_val), id(y_val)]

            self.assertEqual(
                    clean_data_done, # tuple of tuples, strings inside
                    (tuple(x_val), tuple(y_val)), # x_val is list of strings, y_val is list of strings
                    f"Incorrect data cleaning. Expected: {clean_data_done}, got: {(tuple(x_val), tuple(y_val))}"
            )


############################################################
# part 3.2: fitting a quadratic model
############################################################


@testsuite_options(20, 1)
class TestPart3_2(unittest.TestCase):
    """Test fit quadratic model"""

    @case_options(
        5,
        "Your fit_quadratic_model function does not find the correct coefficients for quadratic data",
        "Task test_fit_quadratic_model_small error",
    )
    def test_fit_quadratic_model(self):
        for case_name, case in fit_quadratic_cases.items():
            x = case["x"]
            y = case["y"]
            a_range = case["a_range"]
            b_range = case["b_range"]
            c_range = case["c_range"]
            precision = case["precision"]
            expected = case["expected"]

            a_optimal, b_optimal, c_optimal = pset.fit_quadratic_model(
                x, y, a_range, b_range, c_range, precision
            )

            self.assertAlmostEqual(a_optimal, expected[0], delta=precision * 2,
                             msg=f"Expected a ≈ {expected[0]}, got {a_optimal} for {case_name} case")

            self.assertAlmostEqual(b_optimal, expected[1], delta=precision * 2,
                                msg=f"Expected b ≈ {expected[1]}, got {b_optimal} for {case_name} case")

            self.assertAlmostEqual(c_optimal, expected[2], delta=precision * 2,
                                msg=f"Expected c ≈ {expected[2]}, got {c_optimal} for {case_name} case")

    @case_options(
        2,
        "Your predict_y function does not find the correct y-values for quadratic functions",
        "Task test_predict_y_quadratic error",
    )
    def test_predict_y_quadratic(self):
        quadratic_cases = predict_y_cases["quadratic"]
        for case_name, case in quadratic_cases.items():
            predicted_y = []
            for x in case["x-vals"]:
                predicted_y.append(pset.predict_y(x, case["coefficients"]))

            # check each value in predicted_y against expected_y
            for pred, exp in zip(predicted_y, case["expected_y"]):
                self.assertAlmostEqual(pred, exp, places=3,
                    msg=f"Incorrect predicted y values for {case_name} quadratic case. Expected: {case['expected_y']}, got: {predicted_y}")


############################################################
# part 4: bisection search for regression coefficients
############################################################


@testsuite_options(4, 1)
class TestPart4(unittest.TestCase):
    """Test fit linear bisection model"""

    @case_options(
        5,
        "Your fit_linear_model_bisection function does not find the correct coefficients for linear data",
        "Task test_fit_linear_model_bisection error",
    )
    def test_fit_linear_model_bisection(self):
        for case_name, case in fit_linear_bisect_cases.items():
            a_optimal, b_optimal = pset.fit_linear_model_bisection(
                case["x"], case["y"], case["a_range"], case["precision"]
            )

            self.assertAlmostEqual(a_optimal, case["expected"][0], places=2,
                             msg=f"Expected a ≈ {case['expected'][0]}, got {a_optimal} for {case_name} case")

            self.assertAlmostEqual(b_optimal, case["expected"][1], places=2,
                                msg=f"Expected b ≈ {case['expected'][1]}, got {b_optimal} for {case_name} case")


############################################################
# test results reporting
############################################################


class Results_600(unittest.TextTestResult):
    """Custom test result class to capture output and points."""

    def __init__(self, *args, **kwargs):
        super(Results_600, self).__init__(*args, **kwargs)
        self.output = []
        self.points = 0
        self.max_points = 0

    def _getOptions(self, test):
        method_name = getattr(test, "_testMethodName")
        method = getattr(test, method_name)
        func = method.__func__
        points = getattr(func, "points", 0)
        failure_msg = getattr(func, "failure_message", "")
        error_msg = getattr(func, "error_message", "")
        return points, failure_msg, error_msg

    def addSuccess(self, test):
        points, _, _ = self._getOptions(test)
        self.points += points
        self.max_points += points
        return super().addSuccess(test)

    def addFailure(self, test, err):
        points, failure_msg, _ = self._getOptions(test)
        self.output.append(f"❌ [-{points}] {failure_msg}, {err[1]}\n")
        self.max_points += points
        super().addFailure(test, err)

    def addError(self, test, err):
        points, _, error_msg = self._getOptions(test)
        self.output.append(f"❌ [-{points}] {error_msg}, {err[1]}\n")
        self.max_points += points
        super().addError(test, err)

    def getOutput(self):
        """Return the captured output."""
        if self.points > 0:
            self.output.append(
                f"\n✅ [+{self.points}] "
                f"{'All' if self.points == self.max_points else 'Some'}"
                f" tests passed!\n"
            )
        return "\n".join(self.output)

    def getPoints(self):
        """Return the total points."""
        return self.points


if __name__ == "__main__":
    test_parts = [
        TestPart2_1,
        TestPart2_2,
        TestPart2_3,
        TestPart3_1,
        TestPart3_2,
        TestPart4,
    ]

    suite = unittest.TestSuite()
    for part in test_parts:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(part))
    runner = unittest.TextTestRunner(resultclass=Results_600, verbosity=2)
    result = runner.run(suite)

    output = result.getOutput()
    points_earned = round(result.getPoints(), 3)
    print(output)
    print(f"Total points: {points_earned} / {result.max_points}")
    print(f"Score: {points_earned / result.max_points:4.0%}")
