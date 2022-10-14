# Detection of Lane Lines (Use case: Autonomous Vehicles)

Developed the python script to detect the lane lines in the input images and videos using classic computer vision methods.

## Approach:
1. Input Image

    ![image](https://user-images.githubusercontent.com/62834697/195820963-fb6505b0-703e-4981-af53-ab90ebe31acb.png)

2. Converting the input RGB image to the Grayscale and also removing the noise by applying Gaussian filter

    ![image](https://user-images.githubusercontent.com/62834697/195821581-59da219c-d689-4e7b-bbe5-754a95a40811.png)
    
3. Edge Detection: Apply the canny edge detection algo to enhances the edges in the grayscale image

    ![image](https://user-images.githubusercontent.com/62834697/195821941-16f953a4-d6f4-40aa-bb91-51a1826dd3c2.png)

4. Masking: Masked the region which is not area of interest, then apply the canny edge detector

    ![image](https://user-images.githubusercontent.com/62834697/195822323-44194c02-a987-47ae-8074-232133223ea5.png)

5. Hough Transforms: Detected the lines of interest in the masked image using Hough Transformation.

    ![image](https://user-images.githubusercontent.com/62834697/195822991-530e10e6-c815-49f9-91db-0d065918da17.png)
    
4. Output: 

    ![image](https://user-images.githubusercontent.com/62834697/195823115-2ec5a75d-e63e-44be-891b-e17dbe91ddd5.png)
