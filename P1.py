#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#get_ipython().run_line_magic('matplotlib', 'inline')

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

    gray = grayscale(image)
 
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    low_threshold = 100
    high_threshold = 150
    canny_out = canny(blur_gray, low_threshold, high_threshold)
 
    # give the vertices of polygon in an array form
    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[[100, ysize], [450,325],[525,325], [850,ysize]]], dtype=np.int32)
    #Region of interest
    masked_image = region_of_interest(canny_out, vertices)
 
    #Hough transforms
    img = masked_image
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_len = 10
    max_line_gap = 2
    line_img = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
 
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),min_line_len, max_line_gap)

    pos_slope_line_x = []
    pos_slope_line_y = []
    neg_slope_line_x = []
    neg_slope_line_y = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if((y2-y1)/(x2-x1) > 0):
                pos_slope_line_x.append(x1)
                pos_slope_line_x.append(x2)
                pos_slope_line_y.append(y1)
                pos_slope_line_y.append(y2)
            else:
                neg_slope_line_x.append(x1)
                neg_slope_line_x.append(x2)
                neg_slope_line_y.append(y1)
                neg_slope_line_y.append(y2)
                
    fit_left =  np.polyfit(pos_slope_line_x,pos_slope_line_y,1)
    fit_right = np.polyfit(neg_slope_line_x,neg_slope_line_y,1)
    
    # y =fit_left[0] * x + fit_left[1]
    # ysize = fit_left[0] *x + fit_left[1]
    # x = ysize - fit_left[1] dividedby fit_left[0]
    xpos_bottomleft = (ysize - fit_left[1])/fit_left[0]
    xpos_topleft = (325 - fit_left[1])/fit_left[0]
    xpos_bottomrigt = (ysize - fit_right[1])/fit_right[0]
    xpos_topright = (325 - fit_right[1])/fit_right[0]
    cv2.line(line_img, (int(xpos_bottomleft), ysize), (int(xpos_topleft), 325), 255, 20)
    cv2.line(line_img, (int(xpos_bottomrigt), ysize), (int(xpos_topright), 325), 255, 20)
    if(int(xpos_bottomrigt) > xsize):
        print('bot right = ', xpos_bottomrigt)
        print('pos slope = ', fit_left[0])
    if(int(xpos_topright) < 0):
        print('topright = ', xpos_topright)
        print('neg slope = ', fit_right[0])
        
    #draw_lines(line_img, lines, color=[200, 0, 0], thickness=25)
 
    weighted_image = weighted_img(line_img, image, α=1, β=0.5, γ=0.)
 
    result = weighted_image


    return result
    

import os
images = os.listdir("test_images/")
#image = mpimg.imread('test.jpg')

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
for image in images:
    print('image = ', "test_images/" + image)
    print('This image is:', type(mpimg.imread("test_images/" + image)), 'with dimensions:', mpimg.imread("test_images/" + image).shape)
    output_image = process_image(mpimg.imread("test_images/" + image))
    plt.imshow(output_image)
    plt.show()


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

    gray = grayscale(image)
 
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    low_threshold = 100
    high_threshold = 150
    canny_out = canny(blur_gray, low_threshold, high_threshold)
 
    # give the vertices of polygon in an array form
    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[[100, ysize], [450,325],[525,325], [850,ysize]]], dtype=np.int32)
    #Region of interest
    masked_image = region_of_interest(canny_out, vertices)
 
    #Hough transforms
    img = masked_image
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_len = 10
    max_line_gap = 2
    line_img = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
 
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),min_line_len, max_line_gap)

    pos_slope_line_x = []
    pos_slope_line_y = []
    neg_slope_line_x = []
    neg_slope_line_y = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if((y2-y1)/(x2-x1) > 0):
                pos_slope_line_x.append(x1)
                pos_slope_line_x.append(x2)
                pos_slope_line_y.append(y1)
                pos_slope_line_y.append(y2)
            else:
                neg_slope_line_x.append(x1)
                neg_slope_line_x.append(x2)
                neg_slope_line_y.append(y1)
                neg_slope_line_y.append(y2)
                
    fit_left =  np.polyfit(pos_slope_line_x,pos_slope_line_y,1)
    fit_right = np.polyfit(neg_slope_line_x,neg_slope_line_y,1)
    
    # y =fit_left[0] * x + fit_left[1]
    # ysize = fit_left[0] *x + fit_left[1]
    # x = ysize - fit_left[1] dividedby fit_left[0]
    xpos_bottomleft = (ysize - fit_left[1])/fit_left[0]
    xpos_topleft = (325 - fit_left[1])/fit_left[0]
    xpos_bottomrigt = (ysize - fit_right[1])/fit_right[0]
    xpos_topright = (325 - fit_right[1])/fit_right[0]
    cv2.line(line_img, (int(xpos_bottomleft), ysize), (int(xpos_topleft), 325), 255, 20)
    cv2.line(line_img, (int(xpos_bottomrigt), ysize), (int(xpos_topright), 325), 255, 20)
    if(int(xpos_bottomrigt) > xsize):
        print('bot right = ', xpos_bottomrigt)
        print('pos slope = ', fit_left[0])
    if(int(xpos_topright) < 0):
        print('topright = ', xpos_topright)
        print('neg slope = ', fit_right[0])
        
    #draw_lines(line_img, lines, color=[200, 0, 0], thickness=25)
 
    weighted_image = weighted_img(line_img, image, α=1, β=0.5, γ=0.)
 
    result = weighted_image


    return result
    
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,10)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)