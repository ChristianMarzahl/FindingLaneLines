import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from math import acos
from math import sqrt
from math import pi

counter = 0

def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * ? + img * ? + ?
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, l)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def extrapolate_lines(lines,angle_lower_bound,angle_upper_bound):

    # calc angle for each line 
    lines_with_angles = zip(lines,map(lambda line: np.rad2deg(np.arctan2(line[0][3]-line[0][1],line[0][2]-line[0][0])),lines))     
    filtered_lines = list(filter(lambda x: (x[1] >= angle_lower_bound and angle_upper_bound >= x[1]) ,lines_with_angles))

    # extract x and y coordinates for poly fit 
    x = []
    y = []
    for line in filtered_lines:
        x = np.append(x,line[0][0][0])
        x = np.append(x,line[0][0][2])
        
        y = np.append(y,line[0][0][1])
        y = np.append(y,line[0][0][3])

    if(len(x) == 0 or len(y) == 0):
         return None, None

    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit) 
    
    return fit_fn, (int(min(x)),int(max(x)))

def color_range_mask(image):
    
     # try hsv color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of yellow color in HSV
    lower_yellow = np.array([93,30,147]) #np.array([93,30,147])
    upper_yellow = np.array([102,255,255]) #np.array([124,255,255])
    # Threshold the HSV image to get only yellow colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0,0,153])
    upper_white  = np.array([255,20,255])
    # Threshold the HSV image to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(mask_yellow,mask_white)
    
    return mask
    
def morphologie_test(mask, image):

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, rectKernel)

    plt.imshow(np.concatenate((mask,blackhat), axis=1))
    plt.show()


def filter_mask(mask, image):

    image = image.copy()
    # find all contours in the mask image
    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for (i, c) in enumerate(cnts):

        # compute the area of the contour along with the bounding box
        # to compute the aspect ratio
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the convex hull of the contour, then use the area of the
        # original contour and the area of the convex hull to compute the
        # solidity
        #hull = cv2.convexHull(c)
        #hullArea = cv2.contourArea(hull)

        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

        #if(float(hullArea) > 0):

            #solidity = area / float(hullArea)
                        
            #if (solidity > 0.4 and solidity < 0.8):   

                #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                #cv2.putText(image, str(solidity)[:3] , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 4)

    return image

def process_image_video(image):

    mask = color_range_mask(image)
    mask = cv2.erode(mask,(3,3))

    canny_image = canny(mask,50,150)

    imshape = canny_image.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]*0.45, imshape[0]*0.6), (imshape[1]*0.55, imshape[0]*0.6), (imshape[1],imshape[0])]], dtype=np.int32)
           
    masked_edges = region_of_interest(canny_image,vertices)

    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap) 
    
    result_mask = np.zeros_like(image) 

    angle_lower_bound = 25
    angle_upper_bound = 35
    fit_fn, xRange = extrapolate_lines(lines,angle_lower_bound = angle_lower_bound, angle_upper_bound = angle_upper_bound)    #25 35
    
    if fit_fn is not None and xRange is not None:    
        line = np.array([(imshape[1], int(fit_fn(imshape[1]))), (xRange[0], int(fit_fn(xRange[0])))])
        result_angle = np.rad2deg(np.arctan2(line[0][1]-line[1][1],line[0][0]-line[1][0]))

        if result_angle > angle_lower_bound and result_angle < angle_upper_bound:
            cv2.line(result_mask, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 255), 5)

    angle_lower_bound = -45
    angle_upper_bound = -30
    fit_fn, xRange = extrapolate_lines(lines,angle_lower_bound = angle_lower_bound, angle_upper_bound = angle_upper_bound)
    if fit_fn is not None and xRange is not None:                
        line = np.array([(0,int(fit_fn(0))),(xRange[1], int(fit_fn(xRange[1])))])
        result_angle = np.rad2deg(np.arctan2(line[1][1]-line[0][1],line[1][0]-line[0][0]))

        if result_angle > angle_lower_bound and result_angle < angle_upper_bound:        
            cv2.line(result_mask, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 0), 5)
    
    return  weighted_img(result_mask,image)


def process_image(image):
    
    gray_image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
    # test if color range in hsv space has any advantage 
    #mask = color_range_mask(image)
    #morphologie_test(mask,image)
    #gray_image = cv2.bitwise_and(image,image, mask= mask)

    # smooth image with gaussian blur and calc canny 
    gaussian_image = gaussian_blur(gray_image,5)
    canny_image = canny(gaussian_image,50,150)

    # mask edge image
    imshape = gray_image.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]*0.45, imshape[0]*0.6), (imshape[1]*0.55, imshape[0]*0.6), (imshape[1],imshape[0])]], dtype=np.int32)
           
    masked_edges = region_of_interest(canny_image,vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap) 
    
    result_mask = np.zeros_like(image) 

    fit_fn, xRange = extrapolate_lines(lines,angle_lower_bound = 25, angle_upper_bound = 35)    
    
    if fit_fn is not None and xRange is not None:    
        interpolation_vertices = np.array([(imshape[1],fit_fn(imshape[1])),(xRange[0], fit_fn(xRange[0]))])
        cv2.line(result_mask, (imshape[1], int(fit_fn(imshape[1]))), (xRange[0], int(fit_fn(xRange[0]))), (255, 0, 255), 5)

    fit_fn, xRange = extrapolate_lines(lines,angle_lower_bound = -45, angle_upper_bound = -30)
    if fit_fn is not None and xRange is not None:                
        interpolation_vertices = np.array([(0,fit_fn(0)),(xRange[1], fit_fn(xRange[1]))])
        cv2.line(result_mask, (0, int(fit_fn(0))), (xRange[1], int(fit_fn(xRange[1]))), (255, 0, 0), 5)
    
    return  weighted_img(result_mask,image)


dir_path = "test_images/"
for path in os.listdir(dir_path):

    image = cv2.imread(os.path.join(dir_path,path)) # mpimg.imread(os.path.join(dir_path,path))
        
    hough_image = process_image_video(image)

    #cv2.imwrite(os.path.join("pipeline_images/hsv",path),np.concatenate((image,hough_image), axis=1))

    plt.imshow(np.concatenate((image,hough_image), axis=1))
    plt.show()


# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip

#yellow_output = 'solidYellowLeft_result.mp4'
#clip2 = VideoFileClip('solidYellowLeft.mp4')
#yellow_clip = clip2.fl_image(process_image_video)
#yellow_clip.write_videofile(yellow_output, audio=False)


#from IPython.display import HTML
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(yellow_output))
