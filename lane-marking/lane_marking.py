# importing libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpinmg
import cv2
import numpy as np

def convert_grayscale(image):
        """Returns the grayscale image.

        :param image: RGB image
        :type image: nd.array

        :rtype: nd.array
        :return: grayscale version of the RGB image    
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gaussian_blur(image,kernel_size):
    """Returns the blurred version(removed noise) of the input image
    
    :param image: RGB image
    :type image: nd.array
    :param kernel_size: Square np array side for eg: (5x5)
    :type kernel_size: int

    :rtype: nd.array
    :return: blurred version on the input image
    """
    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)



def canny_edges(image, low, high):
    """Returns the detected edges in the image
    
    :param image: Grayscale image
    :type image: nd.array
    :param low: low threshold value
    :type low: int
    :param high: high threshold value
    :type high: int

    :rtype:nd.array
    :return: detected edges in the grayscale image
    """
    return cv2.Canny(image, low, high)


def region_of_interest(image, vertices):
    """Return the image that contains only pixel value of the region of interest
    
    :param image: Edge pixel image
    :type imgae: nd.array
    :param vertices: desired shaped polygon sides
    :type vertices: 2D numpy array

    :rtype: nd.array
    :returns: masked image that contains only pixel value of the region of the interest
    """

    # create empty mask
    mask = np.zeros_like(image)

    # set the mask color (incorporate both single and 3 channel images)
    if len(image.shape) > 2:
        color_channels = image.shape[2]
        mask_color = (255,)* color_channels
    else:
        mask_color = 255
    
    # define the polygon using vertices, mask_color
    cv2.fillPoly(mask, vertices, mask_color)

    # apply the mask over the input RGB image
    return cv2.bitwise_and(image, mask)


def draw_lines(image, lines, line_color =[255,0,0], line_thickness = 10):
    """Returns image having lines drawn over it
    
    :param image: RGB image
    :type image: nd.array
    :param lines: 3 dim np array whose elements are ([x1,y1,x2,y2])
    :type lines: 3D numpy array
    :param line_color: color of the line, defaults to red [255,0,0]
    :type line_color: list(int)
    :param line_thickness: thickness of the line 
    :type line_thickness: int

    :rtype: None
    :returns: Nothing
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
        
        # draw line on left lane
        cv2.line(image, (upper_left_x, ymin_global),(lower_left_x, ymax_global), line_color, line_thickness)

        # draw line on right lane
        cv2.line(image, (upper_right_x, ymin_global),(lower_right_x, ymax_global), line_color, line_thickness)


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
        """Returns the image with lines drawn over it
        
        :param image = image containg edge pixels
        :type image = nd.array
        :param rho: distance resolution in pixels of Hough Grid
        :type rho: int
        :param theta: angular resolution in radians of the Hough Grid
        :type theta: degress
        :param threshold: minimum no of votes in the hough grid
        :type threshold: int
        :param min_line_len: minimum no of pixels making up a lines
        :type min_line_len: int
        :param: max_line_gap: maximum gal in pixels between connectable line segment
        :type: int

        :rtype: nd.array
        :return: image with hough lines drawn over it
        """
        detected_lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
        line_image = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)
        
        # call draw_lines function

        draw_lines(line_image, detected_lines)
        return line_image



def blend_images(image1, image2, w1=0.7, w2=1, w=0):
    """Returns the blended version of the input images
    
    :param image1: RGB image
    :type image1: nd.array
    :param image2: RGB image
    :type image2: nd.array
    :param w1: weight term defined for image1 defaults to 0.7
    :type w1: float
    :param w2: weight term defined for image2 defaults to 1
    :type w2: float
    :param w: weight term to be added to both image
    :type w: float

    :rtype: nd.array
    :return: blended version of the given 2 input images
    """
    return cv2.addWeighted(image2, w1, image1, w2, w)


def run_lane_detection(image):
    """Returns image showing the lanes in the input image
    
    :param image: RGB image
    :type image: nd.array

    :rtype: RGB image
    :return: image showing the lanes
    """

    # step1 - convert into grayscale
    gray =convert_grayscale(image)

    # step2 - blur the image (remove noise)
    kernel = 5
    blur_image =gaussian_blur(image=image, kernel_size=kernel)

    # step3 - detect edges
    low = 50
    high = 150
    edges = canny_edges(image=gray, low=low, high=high)

    # step4 - mask region : keep only region of interest
    y,x,_ = image.shape
    vertices = np.array([[(0,y),(460,315),(490,315),(x,y)]])
    masked_image =region_of_interest(image=edges, vertices=vertices)

    # step5 - apply hough tranform over the masked image
    rho = 1
    theta = np.pi/180
    threshold = 5
    min_line_length = 55
    max_line_gap = 25
    line_image = np.copy(image)*0
    line_image =hough_lines(image=masked_image, rho=rho, theta=theta, threshold=threshold, min_line_len=min_line_length, max_line_gap=max_line_gap)

    # step6 - blend the images
    color_edges = np.dstack((edges, edges, edges))

    result =blend_images(line_image, image) 
    return result

## can also implement all functions in the form of LaneMarking Class
''''
class LaneMarking:
    """Base class for the Lane Detection/Marking Task"""

    def __init__(self) -> None:
        print("Iniatize the class")
        pass
    
    def convert_grayscale(self, image):
        """Returns the grayscale image.

        :param image: RGB image
        :type image: nd.array

        :rtype: nd.array
        :return: grayscale version of the RGB image    
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    

    def gaussian_blur(self,image,kernel_size):
        """Returns the blurred version(removed noise) of the input image
        
        :param image: RGB image
        :type image: nd.array
        :param kernel_size: Square np array side for eg: (5x5)
        :type kernel_size: int

        :rtype: nd.array
        :return: blurred version on the input image
        """
        return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    
    def canny_edges(self, image, low, high):
        """Returns the detected edges in the image
        
        :param image: Grayscale image
        :type image: nd.array
        :param low: low threshold value
        :type low: int
        :param high: high threshold value
        :type high: int

        :rtype:nd.array
        :return: detected edges in the grayscale image
        """
        return cv2.Canny(image, low, high)

    def region_of_interest(self,image, vertices):
        """Return the image that contains only pixel value of the region of interest
        
        :param image: Edge pixel image
        :type imgae: nd.array
        :param vertices: desired shaped polygon sides
        :type vertices: 2D numpy array

        :rtype: nd.array
        :returns: masked image that contains only pixel value of the region of the interest
        """

        # create empty mask
        mask = np.zeros_like(image)

        # set the mask color (incorporate both single and 3 channel images)
        if len(image.shape) > 2:
            color_channels = image.shape[2]
            mask_color = (255,)* color_channels
        else:
            mask_color = 255
        
        # define the polygon using vertices, mask_color
        cv2.fillPoly(mask, vertices, mask_color)

        # apply the mask over the input RGB image
        return cv2.bitwise_and(image, mask)

    def draw_lines(self, image, lines, line_color =[255,0,0], line_thickness = 10):
        """Returns image having lines drawn over it
        
        :param image: RGB image
        :type image: nd.array
        :param lines: 3 dim np array whose elements are ([x1,y1,x2,y2])
        :type lines: 3D numpy array
        :param line_color: color of the line, defaults to red [255,0,0]
        :type line_color: list(int)
        :param line_thickness: thickness of the line 
        :type line_thickness: int

        :rtype: None
        :returns: Nothing
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
            
            # draw line on left lane
            cv2.line(image, (upper_left_x, ymin_global),(lower_left_x, ymax_global), line_color, line_thickness)

            # draw line on right lane
            cv2.line(image, (upper_right_x, ymin_global),(lower_right_x, ymax_global), line_color, line_thickness)

    def hough_lines(self, image, rho, theta, threshold, min_line_len, max_line_gap):
        """Returns the image with lines drawn over it
        
        :param image = image containg edge pixels
        :type image = nd.array
        :param rho: distance resolution in pixels of Hough Grid
        :type rho: int
        :param theta: angular resolution in radians of the Hough Grid
        :type theta: degress
        :param threshold: minimum no of votes in the hough grid
        :type threshold: int
        :param min_line_len: minimum no of pixels making up a lines
        :type min_line_len: int
        :param: max_line_gap: maximum gal in pixels between connectable line segment
        :type: int

        :rtype: nd.array
        :return: image with hough lines drawn over it
        """
        detected_lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
        line_image = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)
        
        # call draw_lines function

        self.draw_lines(line_image, detected_lines)
        return line_image
    

    def blend_images(self, image1, image2, w1=0.7, w2=1, w=0):
        """Returns the blended version of the input images
        
        :param image1: RGB image
        :type image1: nd.array
        :param image2: RGB image
        :type image2: nd.array
        :param w1: weight term defined for image1 defaults to 0.7
        :type w1: float
        :param w2: weight term defined for image2 defaults to 1
        :type w2: float
        :param w: weight term to be added to both image
        :type w: float

        :rtype: nd.array
        :return: blended version of the given 2 input images
        """
        return cv2.addWeighted(image2, w1, image1, w2, w)
    

    def run_lane_detection(self, image):
        """Returns image showing the lanes in the input image
        
        :param image: RGB image
        :type image: nd.array

        :rtype: RGB image
        :return: image showing the lanes
        """

        # step1 - convert into grayscale
        gray = self.convert_grayscale(image)

        # step2 - blur the image (remove noise)
        kernel = 5
        blur_image = self.gaussian_blur(image=image, kernel_size=kernel)

        # step3 - detect edges
        low = 50
        high = 150
        canny_edges = self.canny_edges(image=gray, low=low, high=high)

        # step4 - mask region : keep only region of interest
        y,x,_ = image.shape
        vertices = np.array([[(0,y),(460,315),(490,315),(x,y)]])
        masked_image = self.region_of_interest(image=canny_edges, vertices=vertices)

        # step5 - apply hough tranform over the masked image
        rho = 1
        theta = np.pi/180
        threshold = 5
        min_line_length = 55
        max_line_gap = 25
        line_image = np.copy(image)*0
        line_image = self.hough_lines(image=masked_image, rho=rho, theta=theta, threshold=threshold, min_line_len=min_line_length, max_line_gap=max_line_gap)

        # step6 - blend the images
        color_edges = np.dstack((canny_edges, canny_edges, canny_edges))

        result = self.blend_images(line_image, image) 
        return result
'''