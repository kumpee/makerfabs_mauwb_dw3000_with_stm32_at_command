"""
File:           plot_2d_map_realtime.py
Author:         Kumpee Teeravech
Version:        2024-06-31
"""
import numpy
import cv2
import csv
import json
import serial
import time
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.animation as animation

def load_csv(filename, delimiter=",", n_skip_lines=-1, n_cols=-1):
    """
    """
    x = []
    k = 0
    with open(filename, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        for row in csv_reader:
            #print(', '.join(row))
            print(row)
            if(k > n_skip_lines):
                x.append([
                    float(row[0]),
                    float(row[1]),])
            k =k + 1
    return x

def process_data(data):
    """
    """
    # unit: centimeter
    x_a0 = numpy.asarray([75.0, 0.0, 0.0])
    x_a1 = numpy.asarray([0.0, 0.0, 0.0])
    x_a2 = numpy.asarray([0.0, 0.0, 0.0])

    range_list = data["range"]
    rssi_list  = data["rssi"]

    #print(range_list)
    #print(rssi_list)

    d_a0 = range_list[0]
    d_a1 = range_list[1]
    d_a2 = range_list[2]

    r_a0 = rssi_list[0]
    r_a1 = rssi_list[1]
    r_a2 = rssi_list[2]

    # check number of available data
    n_anchors = 3
    if(d_a0 == 0.0):
        n_anchors -= 1
    if(d_a1 == 0.0):
        n_anchors -= 1
    if(d_a2 == 0.0):
        n_anchors -= 1
    print("Number of available anchors: {:}".format(n_anchors))

def three_point(x1, y1, x2, y2, r1, r2):
    """ Taken from MakerFabs's MaUWB_DW3000_with_ST_command's github
    """
    temp_x = 0.0
    temp_y = 0.0

    # distance between the two anchors
    p2p = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)
    p2p = numpy.sqrt(p2p)

    # check if the sum of r1 and r2 is less than
    # the distance between the two ancors
    if r1 + r2 <= p2p:
        temp_x = x1 + (x2 - x1) * r1 / (r1 + r2)
        temp_y = y1 + (y2 - y1) * r1 / (r1 + r2)
    else:
        dr = p2p / 2 + (r1 * r1 - r2 * r2) / (2 * p2p)
        temp_x = x1 + (x2 - x1) * dr / p2p
        temp_y = y1 + (y2 - y1) * dr / p2p

    return numpy.asarray([temp_x, temp_y])

def cal_tag_position_trilateration(pt_a0, pt_a1, pt_a2, d_a0, d_a1, d_a2):
    """ Trilateration
    """
    #pt = numpy.zeros(3)

    p_01 = three_point(
            pt_a0[0], pt_a0[1],
            pt_a1[0], pt_a1[1],
            d_a0,
            d_a1)
    p_02 = three_point(
            pt_a0[0], pt_a0[1],
            pt_a2[0], pt_a2[1],
            d_a0,
            d_a2)
    p_12 = three_point(
            pt_a1[0], pt_a1[1],
            pt_a2[0], pt_a2[1],
            d_a1,
            d_a2)

    # average
    p = numpy.asarray([p_01, p_02, p_12])
    p = numpy.mean(p, 0)
    #print(p, p_01, p_02, p_12)

    return p, p_01, p_02, p_12

def update_map(img_map, map_res, map_ext, pts_a, 
                range_list, rssi_list, 
                range_old_list, range_list_raw,
                tag_pos, tag_pos2, tag_pos2_raw,
                pt_others):
    """
    """
    n_anchors = len(pts_a)

    # anchors
    color_a = [
        (  0,   0, 255),
        (  0, 255,   0),
        (255, 100,   0),
    ]

    # STD of measured distances
    color_r_std = [
        ( 20,  20, 60),
        ( 20, 60,  20),
        (80,  20,  20),
    ]

    # raw (un-adjusted) measured distances
    color_r_raw = [
        ( 20,  50, 120),
        ( 50, 120,  50),
        (120,  50,  20),
    ]

    # grid line color
    color_grid = (50, 50, 50)

    font_name  = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # image dimensions
    ih = img_map.shape[0]
    iw = img_map.shape[1]

    map_x_min = map_ext[0]
    map_y_min = map_ext[1]
    map_x_max = map_ext[2]
    map_y_max = map_ext[3]
    map_x_range = map_x_max - map_x_min
    map_y_range = map_y_max - map_y_min
    map_res     = map_x_range/float(iw) # cm/pixel

    # reset map to black
    img_map = img_map * 0

    # draw grid (cm)
    grid_spacing = 50
    grid_x = map_x_min
    grid_y = map_y_min
    while(grid_x <= map_x_max):
        x = iw * ((grid_x - map_x_min)/map_x_range)
        x = int(x)
        cv2.line(
                img_map,
                (x, 0),
                (x, ih),
                color_grid,
                1)
        grid_x = grid_x + grid_spacing
    while(grid_y <= map_y_max):
        y = iw * ((grid_y - map_y_min)/map_y_range)
        y = int(y)
        cv2.line(
                img_map,
                (0, y),
                (iw, y),
                color_grid,
                1)
        grid_y = grid_y + grid_spacing

    # measurement values
    #print(numpy.std(range_old_list, 0))
    for i in range(0, n_anchors, 1):
        # position the i-th anchor
        x = iw * ((pts_a[i][0] - map_x_min)/map_x_range)
        y = ih * ((pts_a[i][1] - map_y_min)/map_y_range)
        x = int(x)
        y = int(y)

        # raw distances (unadjusted)
        r = range_list_raw[i] / map_res
        r = int(r)
        if(r <= 0):
            r = 1
        cv2.circle(
                img_map, 
                (x, y), 
                r, 
                color_r_raw[i], 
                1)

        # adjusted distances
        r = range_list[i] / map_res
        r = int(r)
        if(r <= 0):
            r = 1
        cv2.circle(
                img_map, 
                (x, y), 
                r, 
                (int(0.8*color_a[i][0]), int(0.8*color_a[i][1]), int(0.8*color_a[i][2])),
                1)

        # uncertainty : std
        r_std = numpy.std(range_old_list[:, i])
        r_std = r_std / map_res
        r_std = int(r_std)
        if(r_std <= 0):
            r_std = 1
        r_min = r - r_std
        r_max = r + r_std
        if(r_min <= 0):
            r_min = 1
        if(r_max <= 0):
            r_max = 1
        cv2.circle(
                img_map, 
                (x, y), 
                r_min, 
                color_r_std[i], 
                1)
        cv2.circle(
                img_map, 
                (x, y), 
                r_max, 
                color_r_std[i], 
                1)

    # anchors
    for i in range(0, n_anchors, 1):
        x = iw * ((pts_a[i][0] - map_x_min)/map_x_range)
        y = ih * ((pts_a[i][1] - map_y_min)/map_y_range)
        x = int(x)
        y = int(y)
        r = 9
        cv2.circle(
                img_map, 
                (x, y), 
                r, 
                color_a[i],
                -1)
        cv2.circle(
                img_map, 
                (x, y), 
                r + 3, 
                (255,255,255),
                2)
        cv2.putText(
                img_map,
                "A{:1d}".format(i),
                (x+r+5, y+r+3),
                font_name,
                font_scale,
                color_a[i],
                font_thickness,
                cv2.LINE_AA)

    # tags
    if(tag_pos is not None):
        for i in range(1, len(tag_pos), 1):
            x = iw * ((tag_pos[i][0] - map_x_min)/map_x_range)
            y = ih * ((tag_pos[i][1] - map_y_min)/map_y_range)
            x = int(x)
            y = int(y)
            cv2.circle(
                    img_map, 
                    (x, y), 
                    3, 
                    color_a[i-1], 
                    -1)
        x = iw * ((tag_pos[0][0] - map_x_min)/map_x_range)
        y = ih * ((tag_pos[0][1] - map_y_min)/map_y_range)
        x = int(x)
        y = int(y)
        cv2.circle(
                img_map, 
                (x, y), 
                7, 
                (255,255,255), 
                2)
        cv2.putText(
                img_map,
                "T_adj_tri",
                (x+8, y),
                font_name,
                font_scale,
                (255,255,255), 
                font_thickness,
                cv2.LINE_AA)

    # tag-leastsquare method (adjusted by LS coefficients)
    if(tag_pos2 is not None):
        x = iw * ((tag_pos2[0] - map_x_min)/map_x_range)
        y = ih * ((tag_pos2[1] - map_y_min)/map_y_range)
        x = int(x)
        y = int(y)
        tag_color = (0, 255, 255)
        cv2.circle(
                img_map, 
                (x, y), 
                7, 
                tag_color, 
                -1)
        cv2.putText(
                img_map,
                "T_adj_lsq",
                (x+8, y),
                font_name,
                font_scale,
                tag_color, 
                font_thickness,
                cv2.LINE_AA)
        
        # distances to anchors
        xt = x
        yt = y
        for i in range(0, 3, 1):
            xa = iw * ((pts_a[i][0] - map_x_min)/map_x_range)
            ya = ih * ((pts_a[i][1] - map_y_min)/map_y_range)

            xc = (xt + xa)/2.0
            yc = (yt + ya)/2.0

            d = numpy.sqrt((xt-xa)**2 + (yt-ya)**2)
            cv2.line(
                img_map,
                (int(xt), int(yt)),
                (int(xa), int(ya)),
                color_a[i],
                1)
            cv2.putText(
                img_map,
                "{:.2f}".format(d),
                (int(xc), int(yc)),
                font_name,
                font_scale,
                (255,255,255), #color_a[i],
                font_thickness,
                cv2.LINE_AA)

    # tag-leastsquare method - raw measurement data
    if(tag_pos2_raw is not None):
        x = iw * ((tag_pos2_raw[0] - map_x_min)/map_x_range)
        y = ih * ((tag_pos2_raw[1] - map_y_min)/map_y_range)
        x = int(x)
        y = int(y)
        tag_color = (255, 0, 255)
        cv2.circle(
                img_map, 
                (x, y), 
                7, 
                tag_color, 
                -1)
        cv2.putText(
                img_map,
                "T_raw_lsq",
                (x+8, y),
                font_name,
                font_scale,
                tag_color, 
                font_thickness,
                cv2.LINE_AA)

    # other ground truth points
    if(pt_others is not None):
        r = 7
        tag_color = (255, 255, 0)
        for i in range(0, len(pt_others), 1):
            x = iw * ((pt_others[i][0] - map_x_min)/map_x_range)
            y = ih * ((pt_others[i][1] - map_y_min)/map_y_range)
            x = int(x)
            y = int(y)
            tag_color = (255, 0, 255)
            cv2.circle(
                    img_map, 
                    (x, y), 
                    r, 
                    tag_color, 
                    2)

    #cv2.imshow("img_map", img_map)
    #u_key = cv2.waitKey(1) # milliseconds 
    #if(u_key == ord('q')):
    #    break
    return img_map

def cal_tag_position_my_leastsquare(anchor1, anchor2, anchor3, d1, d2, d3):
    """ Estimate position of a tag given distances to anchors and positions
        of the anchors using non-linear least squares methods.
        @see https://www.researchgate.net/publication/344188953_UWB_indoor_localization_using_deep_learning_LSTM_networks
    """
    x1 = anchor1[0]
    y1 = anchor1[1]
    x2 = anchor2[0]
    y2 = anchor2[1]
    x3 = anchor3[0]
    y3 = anchor3[1]

    #A = [
    #    2*(a2 - a1),
    #    2*(a3 - a1)]
    A = numpy.asarray([
        [2*(x2-x1), 2*(y2-y1)],
        [2*(x3-x1), 2*(y3-y1)],
    ])

    # squared
    x1_sq = x1**2
    y1_sq = y1**2
    x2_sq = x2**2
    y2_sq = y2**2
    x3_sq = x3**2
    y3_sq = y3**2
    d1_sq = d1**2
    d2_sq = d2**2
    d3_sq = d3**2
    b = numpy.asarray([
        [d1_sq - d2_sq + x2_sq + y2_sq - x1_sq - y1_sq],
        [d1_sq - d3_sq + x3_sq + y3_sq - x1_sq - y1_sq],
    ])

    try:
        p, _, _, _ = numpy.linalg.lstsq(A, b, rcond=None)
    except numpy.linalg.LinAlgError:
        # No solution found
        p = None

    return p

def main():
    """
    """
    # coefficients
    #coeff = load_csv("../data/coeff.txt", ",", -1)
    coeff = load_csv("../data/coef_lstsq.txt", ",", -1, n_cols=-1)

    # Output CSV file
    filename_out = "../data/log_realtime_{:}.txt".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    fs_out = open(filename_out, "w")

    # commmuniction port
    port_name = "COM13"
    baud_rate = "115200"
    ser = serial.Serial(port_name, baud_rate, timeout=0.1)
    
    # ground truth positions of the ancors (cm)
    pt_a0 = numpy.asarray([ 800.0,  800.0, 0.0])
    pt_a1 = numpy.asarray([ 800.0,    0.0, 0.0])
    pt_a2 = numpy.asarray([   0.0,  400.0, 0.0])
    pts_a = [pt_a0, pt_a1, pt_a2]

    # other ground truth positions
    pt_others = numpy.asarray([
        [      255.0,       247.0],
        [800.0-175.0,       252.0],
        [      253.0, 800.0-240.0],
        [800.0-180.0, 800.0-234.0],
    ])

    # image
    map_x_min = -50.0
    map_y_min = -50.0
    map_x_max = 900.0
    map_y_max = 900.0
    map_ext = [map_x_min, map_y_min, map_x_max, map_y_max]

    # map canvas
    canvas_size = 1000
    img_map = numpy.zeros((canvas_size, canvas_size, 3), dtype="uint8")
    map_res = map_x_max/float(canvas_size) # pixel resolution (cm/pixel)

    # measurement uncertainty
    n_old_data     = 10
    range_old_list = numpy.zeros((n_old_data, 3))
    range_old_temp = numpy.zeros((n_old_data, 3))

    time_max   = 5000 # seconds
    time_start = time.time() # seconds
    is_started = False
    print('Waiting for UWB messages,...')
    while(True):
        try:
            msg = ser.readline().decode().strip()
        except Exception as e:
            msg = None

        data = None
        if msg:
            try:
                data = json.loads(msg)
            except Exception as e:
                data = None

        if data:
            if(is_started == False):
                is_started = True
                time_start = time.time() # seconds

            # check elapsed time
            time_current = time.time()
            time_elapsed = time_current - time_start
            if(time_elapsed >= time_max):
                break

            # get the ranges
            range_list     = data["range"]
            rssi_list      = data["rssi"]
            range_list_raw = range_list.copy()

            # append data to file
            fs_out.write("{:},{:},{:},{:},{:},{:}\n".format(
                    range_list[0],
                    range_list[1],
                    range_list[2],
                    rssi_list[0],
                    rssi_list[1],
                    rssi_list[2],
            ))

            # adjust measurement data using coefficients calculated by least squares method
            # x' = mx + b
            for i in range(0, 3, 1):
                range_list[i] = (coeff[i][0]*range_list[i]) + coeff[i][1]

            # update old data queue
            range_old_temp = range_old_list.copy()
            range_old_list[0:n_old_data-1, :] = range_old_temp[1:n_old_data, :]
            range_old_list[n_old_data-1, 0]   = range_list[0]
            range_old_list[n_old_data-1, 1]   = range_list[1]
            range_old_list[n_old_data-1, 2]   = range_list[2]
            
            print("[{:>6.2f}],{:},{:},{:},{:},{:},{:}".format(
                    time_elapsed,
                    range_list[0],
                    range_list[1],
                    range_list[2],
                    rssi_list[0],
                    rssi_list[1],
                    rssi_list[2],
            ))

            # Calculate the position (LS method)
            tag_pos2_raw = cal_tag_position_my_leastsquare(
                    pt_a0, 
                    pt_a1, 
                    pt_a2, 
                    range_list_raw[0], # raw distance
                    range_list_raw[1], # raw distance
                    range_list_raw[2]) # raw distance

            # Calculate the position (LS method)
            tag_pos2 = cal_tag_position_my_leastsquare(
                    pt_a0, 
                    pt_a1, 
                    pt_a2, 
                    range_list[0], # distance adjusted by the coefficients
                    range_list[1], # distance adjusted by the coefficients
                    range_list[2]) # distance adjusted by the coefficients

            # Calculate the position (trileration)
            p, p_01, p_02, p_12 = cal_tag_position_trilateration(
                    pt_a0, 
                    pt_a1, 
                    pt_a2, 
                    range_list[0]/1.0, # distance adjusted by the coefficients
                    range_list[1]/1.0, # distance adjusted by the coefficients
                    range_list[2]/1.0) # distance adjusted by the coefficients
            tag_pos = [p, p_01, p_02, p_12]

            # show map
            img_map = update_map(
                        img_map,
                        map_res,
                        map_ext,
                        pts_a,
                        range_list,
                        rssi_list,
                        range_old_list,
                        range_list_raw,
                        tag_pos,
                        tag_pos2,
                        tag_pos2_raw,
                        pt_others)
            
            cv2.imshow("img_map", img_map)
            u_key = cv2.waitKey(1) # milliseconds 
            if(u_key == ord('q')):
                break

    # done
    ser = None
    cv2.destroyAllWindows()
    fs_out.close()

if __name__ == "__main__":
    main()