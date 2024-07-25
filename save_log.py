"""
File:           save_log.py
Author:         Kumpee Teeravech
Version:        2024-06-31
"""

import numpy
import cv2
import json
import serial
import time
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.animation as animation

def main():
    """
    """
    time_max   = 180 # seconds
    
    # commmuniction port
    port_name = "COM13"
    baud_rate = "115200"
    ser = serial.Serial(port_name, baud_rate, timeout=0.1)
    
    # 2024-04-05
    #d_a0 = 56  # cm
    #d_a1 = 142 # cm
    #d_a2 = 114 # cm
    # 2024-04-06
    #d_a0 = 3000 # cm
    #d_a1 = 3000 # cm
    #d_a2 = 3000 # cm
    # 2024-04-07
    #d_a0 = 5000 # cm
    #d_a1 = 5000 # cm
    #d_a2 = 5000 # cm
    # 2024-04-09
    d_a0 = 900 # cm
    d_a1 = 900 # cm
    d_a2 = 900 # cm
    d_min = numpy.min([d_a0, d_a1, d_a2])
    d_max = numpy.max([d_a0, d_a1, d_a2])

    # output flename
    filename_out = "../data/log_test_dist_{:}_{:04d}_{:04d}_{:04d}.txt".format(
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            int(d_a0),
            int(d_a1),
            int(d_a2),)
    f_out = open(filename_out, "w")

    time_start = time.time() # seconds

    while(True):
        # check elapsed time
        time_current = time.time()
        time_elapsed = time_current - time_start
        if(time_elapsed >= time_max):
            break

        # get the string message
        try:
            msg = ser.readline().decode().strip()
        except Exception as e:
            msg = None

        # extract json
        data = None
        if msg:
            try:
                data = json.loads(msg)
            except Exception as e:
                data = None

        # write the data
        if data:
            range_list = data["range"]
            rssi_list  = data["rssi"]

            # write to file
            f_out.write("{:},{:},{:},{:},{:},{:}\n".format(
                    range_list[0],
                    range_list[1],
                    range_list[2],
                    rssi_list[0],
                    rssi_list[1],
                    rssi_list[2],
            ))

            print("[{:>6.2f}],{:},{:},{:},{:},{:},{:}".format(
                    time_elapsed,
                    range_list[0],
                    range_list[1],
                    range_list[2],
                    rssi_list[0],
                    rssi_list[1],
                    rssi_list[2],
            ))

    # done
    f_out.close()

if __name__ == "__main__":
    main()