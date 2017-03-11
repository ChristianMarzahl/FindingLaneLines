
## Reflection

### 1. My pipeline consisted of the following steps. 

1. Make use of the HSV color space to identify the lane lines based on color range threshold   
The yellow lines are in range [93,30,147] to [102,255,255]
The with lines are in range [0,0,153] to [255,20,255]

![solidYellowCurve2](https://drive.google.com/uc?export=view&id=0B7o_AoWlbJt-eG1YWlNjTG9lejA)

2. Calculated the edges with the canny edge detection (low_threshold = 50, high_threshold = 150)
3. Mask the image in the area of the road ahead and calculated the probabilistic Hough Lines  

| Parameter     | Value         | Description|
| ------------- |:-------------:| :-----|
| rho      | 2 | distance resolution in pixels of the Hough grid |
| theta      | PI/180      |   angular resolution in radians of the Hough grid |
| threshold | 15      |    minimum number of votes (intersections in Hough grid cell) |
| min_line_length| 40      |    minimum number of pixels making up a line |
| max_line_gap| 20      |    maximum gap in pixels between connectable line segments |

![solidWhiteCurve](https://drive.google.com/uc?export=view&id=0B7o_AoWlbJt-U3lIOUhUNFFSNlE)

4. Sort the found line segments by angle.   
Left line segment angle range ( angle_lower_bound = -42°, angle_upper_bound = -35°) 
Right line segment angle range (lines,angle_lower_bound = 27°, angle_upper_bound = 32°)

5. Fit the sorted line segments by a first order polynomial  
6. Draw the fitted polynomial  line in range of the road ahead


![solidWhiteCurve](https://drive.google.com/uc?export=view&id=0B7o_AoWlbJt-dDRUOEltY2o0VnM)

### 2. Helper programs to select HSV color space border values

To find a appropriate HSV color space I wrote a helper program. 

![Helper program](https://drive.google.com/uc?export=view&id=0B7o_AoWlbJt-Z0Juc25YUHNKcTA)


### 3. Some of the potential shortcomings of my current pipeline are

1. The mask used to identify the area of interest on the road ahead is not flexible enough for sharp curves
2. Changing light conditions could lead to soft edges, which could be missed by the canny edge detection
3. Changing light conditions could lead wrong segmentation by the HSV in range function 
4. Sharp curves will be missed by the hough line detection  

  

### 4. Possible pipeline improvements could be 

1. To enlarge the area of interest on the road ahead and filter objects that are not lane lines a primitive shape and texture analysis could be used. Has the found lane line candidate a rectangular shape? Has the found lane line candidate a uniform texture and color? 
 

### 5. Results

1. [solidWhiteRight.mp4](https://drive.google.com/uc?export=view&id=0B7o_AoWlbJt-NFVQZXJtM3B2azg)
2. [solidYellowLeft_result.mp4](https://drive.google.com/open?id=0B7o_AoWlbJt-WDJMZnJrQk4zWTg)
3. [Challenge.mp4](https://drive.google.com/open?id=0B7o_AoWlbJt-OEQyalBYd0hzU1E)

4. [Argumentation debug images](https://drive.google.com/drive/folders/0B7o_AoWlbJt-Mkl1ZlNEUDE3SVk?usp=sharing)
5. [Hough debug images](https://drive.google.com/drive/folders/0B7o_AoWlbJt-YkQxUkhrWmJ1emc?usp=sharing)
6. [hsv debug images](https://drive.google.com/drive/folders/0B7o_AoWlbJt-NlhaZ0lNZC1vS1k?usp=sharing)

### 6. Aditional Help

1. [Lane line improving](http://stackoverflow.com/questions/36598897/python-and-opencv-improving-my-lane-detection-algorithm)