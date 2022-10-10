## importing libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



def convert_grayscale(image):
    """Convert the colored image into grayscale

    Input: RGB Image size(HxWxC)
    
    Returns: Grayscale image size(HxWx1)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_canny(image,low_thres, high_thres):
    """Apply the canny edge detector to the input image
    
    Input: 
        Smoothed RGB image

    Parameters: 
        low_thres: low threshold value for edge detection
        high_thres: high threshold value for edge detectiob
    
    Returns:
        Edges present in the image
    """
    return cv2.Canny(image, low_thres, high_thres)


def gaussain_blur(image, kernel_size):
    """Apply the gaussian smoothing over the input image'
    
        Input:
            Input Image: RGB image
        
        Parameters:
            kernel size: will define the square kernel matrix for image smoothing
        
        Returns:
            Smoothed image(less noise)
    """

    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)

def region_of_interest(image, vertices):
    """Mask out the non-interest region with black pixel
    
    Input:
        RGB image
    
    Parameters:
        vetices: sides of desired polygon in form of 2-dimensional numpy array
    
    Returns:
        Masked out image
    """
    # create mask image
    mask = np.zeros_like(image)

    # define the 3 channel or 1 channel color vector to define the mask
    if len(image.shape) > 2:
        channels = image.shape[2]
        mask_color = (255,)* channels
    else:
        mask_color = 255
    
    # fill the polygon defined by vertices with mask_color
    cv2.fillPoly(mask, vertices, mask_color)

    # return the input image where only mask image pixel value are non-zero
    return cv2.bitwise_and(image, mask)


def draw_lines(image, lines, line_color = [255,0,0], line_thickness= 3):
    """Draw the lines over the input image
    
    Input:
        RGB image
    
    Parameters:
        lines: 3 dimensional numpy array whose elements are ([x1,y1,x2,y2]) start and end points of lines
    
    Returns:
        Image with lines drawn over it
    """
    ymin_global = image.shape[0]
    ymax_global = image.shape[0]
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    for line in lines:
            for x1,y1,x2,y2 in line:
                gradient = (y2-y1)/(x2-x1)
                ymin_global = min(min(y1, y2), ymin_global)
                if gradient >0:
                    all_left_grad.append(gradient)
                    all_left_y += [y1,y2]
                    all_left_x += [x1,x2]
                elif gradient <0:
                    all_right_grad.append(gradient)
                    all_right_y += [y1,y2]
                    all_right_x += [x1,x2]
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)                

    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(image, (upper_left_x, ymin_global), 
                      (lower_left_x, ymax_global), line_color, line_thickness)
        cv2.line(image, (upper_right_x, ymin_global), 
                      (lower_right_x, ymax_global), line_color, line_thickness)


def hough_line_tranform(image, rho, theta, threshold, min_line_len, max_line_gap):
    """Apply the hough tranform over the input image
    
    Input:
        RGB image
    
    Parameters:
        rho: distance resolution in pixels of the Hough Grid
        theta: angular resolution in radians of the Hough Grid
        threshold: minimum no of votes in the hough grid (to declare as line)
        min_line_len: minimum no of pixels making up a lines
        max_line_gap: maximum gal in pixels between connectable line segments
    
    Retuns:
        Image with hough lines drawn over it
    """

    lines = cv2.HoughLinesP(image,rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    line_image = np.zeros_like(image)
    draw_lines(line_image,lines)
    return line_image

def blend_two_images(image1,image2, w1=0.7, w2=1, w=0):
    """Blend the two images as per defined weight terms into a single image
    
    Input: 
        Two RGB images

    Parameters:
        w1: weight defined on first image
        w2: weight defined on second image
        w: additional weight added images
    
    Returns:
        Weighted image: image*w1 + image2*w2 + w
    """
    return cv2.addWeighted(image1, w1, image2, w2, w)


def image_processing(image):
    grayscale = convert_grayscale(image)

    kernel_size = 5
    smooth_image = gaussain_blur(image,kernel_size)

    canny_edges = apply_canny(smooth_image,50,150)

    vertices = np.array([[(0,image[0]),(460, 315), (490, 315), (image[1],image[0])]], dtype=np.int32)
    masked_image = region_of_interest(canny_edges,vertices)

    line_image = hough_line_tranform(masked_image, 1, np.pi/180, 5, 55, 25)
    line_edges = blend_two_images(line_image, np.dstack((canny_edges,canny_edges, canny_edges)))

    result = blend_two_images(line_image,image)

    return result


