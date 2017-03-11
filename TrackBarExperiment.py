import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from math import acos
from math import sqrt
from math import pi

def nothing(x):
    pass

dir_path = "test_images/"
#dir_path = "film/"
for path in os.listdir(dir_path):

    image = cv2.imread(os.path.join(dir_path,path))
    #image = cv2.imread(os.path.join(dir_path,"148_.png"))

    window_name = "HSV_Test"
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

    # create trackbars for color change
    cv2.createTrackbar('H_lower',window_name,0,255,nothing)
    cv2.createTrackbar('S_lower',window_name,0,255,nothing)
    cv2.createTrackbar('V_lower',window_name,0,255,nothing)
    
    cv2.createTrackbar('H_upper',window_name,0,255,nothing)
    cv2.createTrackbar('S_upper',window_name,0,255,nothing)
    cv2.createTrackbar('V_upper',window_name,0,255,nothing)

    while(1):

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        h_lower = cv2.getTrackbarPos('H_lower',window_name)
        s_lower = cv2.getTrackbarPos('S_lower',window_name)
        v_lower = cv2.getTrackbarPos('V_lower',window_name)

        h_upper = cv2.getTrackbarPos('H_upper',window_name)
        s_upper = cv2.getTrackbarPos('S_upper',window_name)
        v_upper = cv2.getTrackbarPos('V_upper',window_name)


        # define range of yellow color in HSV
        lower_yellow = np.array([h_lower,s_lower,v_lower])
        upper_yellow = np.array([h_upper,s_upper,v_upper])
        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask_image =  cv2.bitwise_and(image,image, mask= mask)
        
        cv2.imshow(window_name, mask_image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


cv2.destroyAllWindows()