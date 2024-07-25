"""
File:           calibrate_offsets.py
Author:         Kumpee Teeravech
Version:        2024-06-31
"""

import numpy
import cv2
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from scipy import stats

def load_csv(filename, delimiter=",", n_skip_lines=-1):
    """
    """
    x = []
    k = 0
    with open(filename, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        for row in csv_reader:
            if(k > n_skip_lines):
                x.append([
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),])
            k =k + 1
    return x

def load_data(filenames, n_skip_lines=-1):
    data = []
    n_rows = 0

    for i in range(0, len(filenames), 1):
        x = load_csv(
                "../data/{:}".format(filenames[i][1]), 
                ",",
                n_skip_lines)
        data.append(numpy.asarray(x))
        n_rows = n_rows + len(x)
        print("No. of rows: {:}".format(len(x)))
    print("Total number of rows: {:}".format(n_rows))

    return data

def func_linear_regression(x):
    return (slope * x) + intercept

def show_boxplot(x, y, target_list, anchor_id):
    """
    """
    plt.figure(figsize=(12, 12))
    plt.clf()

    range_min = numpy.min(target_list)
    range_max = numpy.max(target_list)

    data   = []
    labels = []
    for i in range(0, len(target_list), 1):
        target_d = target_list[i]
        idx = x == target_d
        xx = x[idx]
        yy = y[idx]
        data.append(yy)
        labels.append("d_{:.0f}".format(target_d))
    plt.boxplot(data, patch_artist=False, labels=labels)

    plt.xlabel("Ground truth distances (cm)")
    plt.ylabel("Measured distances (cm)")
    plt.grid()
    plt.title("Comparison of measurement values vs reference values of Anchor-{:}".format(anchor_id))
    plt.tight_layout()
    plt.show()

def main():
    # params
    n_skip_lines = 10

    # A list of input filenames for the calibraton
    filenames = [
        [0, "log_test_dist_2024_03_31_08_20_14_0100_0100_0100.txt"],
        [0, "log_test_dist_2024_03_31_08_25_04_0200_0200_0200.txt"],
        [0, "log_test_dist_2024_03_31_08_27_32_0200_0200_0200.txt"],
        [0, "log_test_dist_2024_03_31_08_30_55_0300_0300_0300.txt"],
        [0, "log_test_dist_2024_03_31_08_33_40_0300_0300_0300.txt"],
        [0, "log_test_dist_2024_03_31_08_37_53_0400_0400_0400.txt"],
        [0, "log_test_dist_2024_03_31_08_40_37_0400_0400_0400.txt"],
        [0, "log_test_dist_2024_03_31_08_44_25_0500_0500_0500.txt"],
        [0, "log_test_dist_2024_03_31_08_46_51_0500_0500_0500.txt"],
        [0, "log_test_dist_2024_03_31_21_40_32_0050_0050_0050.txt"],
        [0, "log_test_dist_2024_03_31_21_43_10_0050_0050_0050.txt"],
        [0, "log_test_dist_2024_03_31_21_46_09_0100_0100_0100.txt"],
        [0, "log_test_dist_2024_04_01_13_38_02_0100_0100_0100.txt"],
        [0, "log_test_dist_2024_04_01_13_40_38_0100_0100_0100.txt"],
        [0, "log_test_dist_2024_04_01_13_43_21_0200_0200_0200.txt"],
        [0, "log_test_dist_2024_04_01_13_46_02_0200_0200_0200.txt"],
        [0, "log_test_dist_2024_04_01_13_49_07_0300_0300_0300.txt"],
        [0, "log_test_dist_2024_04_01_13_51_49_0300_0300_0300.txt"],
        [0, "log_test_dist_2024_04_01_13_55_31_0400_0400_0400.txt"],
        [0, "log_test_dist_2024_04_01_13_58_27_0400_0400_0400.txt"],
        [0, "log_test_dist_2024_04_01_14_01_02_0500_0500_0500.txt"],
        [0, "log_test_dist_2024_04_01_14_03_41_0500_0500_0500.txt"],
        [0, "log_test_dist_2024_04_01_14_06_33_1000_1000_1000.txt"],
        [0, "log_test_dist_2024_04_01_14_09_17_1000_1000_1000.txt"],
        [0, "log_test_dist_2024_04_01_14_15_21_1500_1500_1500.txt"],
        [0, "log_test_dist_2024_04_01_14_18_06_1500_1500_1500.txt"],
        [0, "log_test_dist_2024_04_01_14_21_11_2000_2000_2000.txt"],
        [0, "log_test_dist_2024_04_01_14_24_28_2000_2000_2000.txt"],
        [0, "log_test_dist_2024_04_01_14_27_56_2500_2500_2500.txt"],
        [0, "log_test_dist_2024_04_01_14_30_45_2500_2500_2500.txt"],
        [0, "log_test_dist_2024_04_01_14_34_19_3000_3000_3000.txt"],
        [0, "log_test_dist_2024_04_01_14_37_13_3000_3000_3000.txt"],
    ]
    n_data = len(filenames)

    # update range from filename
    ranges = numpy.zeros(n_data)
    for i in range(0, n_data, 1):
        temp = filenames[i][1].split("_")
        d = float(temp[len(temp)-2])
        ranges[i] = d
        filenames[i][0] = d
    ranges_unique = numpy.sort(numpy.unique(ranges))
    n_ranges_unique = len(ranges_unique)
    d_min = numpy.min(ranges)
    d_max = numpy.max(ranges)

    print("No. of input files:               {:}".format(n_data))
    print("Total no. of reference distances: {:}".format(n_ranges_unique))
    print("Minimum reference distance:       {:}".format(d_min))
    print("Maximum reference distance:       {:}".format(d_max))

    # load the input data into an array
    data = load_data(filenames, n_skip_lines)

    # -------------------------------------------------------------------------
    # Calculate the coefficients
    # -------------------------------------------------------------------------
    coef       = numpy.zeros((3, 2))
    coef_lstsq = numpy.zeros((3, 2))
    for k in range(0, 3, 1):
        # data buffers
        xx = numpy.zeros(10000) # reference values
        yy = numpy.zeros(10000) # measurement values
        n  = 0
        id1 = 0
        id2 = 0
        for i in range(0, n_data, 1):
            x = ranges[i]
            y = data[i]
            n_y = y.shape[0]
            
            id2 = id1 + n_y
            n   = n + n_y

            xx[id1:id2] = x
            yy[id1:id2] = y[:, k]

            id1 = id2

        xx = xx[0:n]
        yy = yy[0:n]

        # show boxplot
        show_boxplot(
                xx, 
                yy, 
                ranges_unique, 
                k)

        # estimate the coefficients using the least squares technique
        X = numpy.vstack((
                yy, 
                numpy.ones_like(yy))).T
        coefficients, _, _, _ = numpy.linalg.lstsq(
                X, 
                xx, 
                rcond=None)

        # estimate the coefficients using the polyfit method (linear)
        m, b = numpy.polyfit(
                xx,
                yy,
                1)
        coef[k, 0] = m
        coef[k, 1] = b

        coef_lstsq[k, 0] = coefficients[0]
        coef_lstsq[k, 1] = coefficients[1]

    # -------------------------------------------------------------------------
    # Save the coefficents
    # -------------------------------------------------------------------------
    fs_out = open("../data/coeff.txt", "w")
    for i in range(0, 3, 1):
        fs_out.write("{:.6f},{:.6f}\n".format(coef[i][0], coef[i][1]))
    fs_out.close()

    fs_out = open("../data/coef_lstsq.txt", "w")
    for i in range(0, 3, 1):
        fs_out.write("{:.6f},{:.6f}\n".format(coef_lstsq[i][0], coef_lstsq[i][1]))
    fs_out.close()

if __name__ == "__main__":
    main()