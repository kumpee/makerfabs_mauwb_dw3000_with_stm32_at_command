# Overview
These are the Python scripts I used during my review of Makerfabs' 'MaUWB_DW3000 with STM32 AT Command' modules.

![Example screenshot](https://www.cnx-software.com/wp-content/uploads/2024/04/mauwb_dw3000_st_calibration_fig_meeting_room_empty_room_realtime-699x720.png)

# Dependencies
- opencv
- json
- pyserial

# save_log.py
The script saves the data from the UWB device to a comma-delimited text file. In my test, I logged the data from a tag.

# calibrate_offsets.py
This script estimates coefficients (linear) to adjust the offset between the measurement values and the reference values. The coefficients in the `coef` array are estimated by NumPy's `polyfit` function, while the coefficients in `coef_lstsq` are estimated using NumPy's `linalg.lstsq` function. These coefficients are saved as a text file for later use with the `plot_2d_map_realtime.py` script.

# plot_2d_map_realtime.py
This script loads coefficients from a file and extracts real-time measurement data from the serial port. The ground truth coordinates of the three anchors are stored in the `pt_a0`, `pt_a1`, and `pt_a2` variables. The raw measurement values are adjusted using the coefficients. The coordinates of the tag are calculated using the LS method and the trilateration method. I use OpenCV to visualize the calculations.

# Resources:
- MaUWB_DW3000 with STM32 AT Command at [Makerfabs store](https://www.makerfabs.com/mauwb-dw3000-with-stm32-at-command.html)
- My review of MaUWB_DW3000 with STM32 AT Command at [CNX-Software](https://www.cnx-software.com/2024/04/16/mauwb_dw3000-with-stm32-at-command-review-arduino-uwb-range-precision-indoor-positioning/).
- Testing video-1 at [YouTube](https://www.youtube.com/watch?v=i9xFhcEHBYI)
- Testing video-2 at [YouTube](https://www.youtube.com/watch?v=YJQwljjFePU)
- Testing video-3 at [YouTube](https://www.youtube.com/watch?v=M4dDp27HrYc) 
