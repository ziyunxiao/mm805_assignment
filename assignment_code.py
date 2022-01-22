import numpy as np
import cv2
from pprint import pprint
from scipy import ndimage
from skimage.feature import corner_peaks


def imfilter(I, filter)->np.ndarray:
    I_f = ndimage.filters.correlate(I, weights=filter, mode='constant')
    return I_f


def get_harris_points(I, k=0.05):

    # check image and convert to gray and normalize
    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # Step 1 calcualte Axx, Axy and Ayy
    
    # Step 1.1 calculate Ix, Iy
    # apply soble filter for quick calculation
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / \
        8.0  # sobel filter for x derivative
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / \
        8.0  # sobel filter for y derivative

    Ix = imfilter(I, filter_x)
    Iy = imfilter(I, filter_y)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    window = np.ones((3, 3))

    Axx = imfilter(Ixx, window)
    Axy = imfilter(Ixy, window)
    Ayy = imfilter(Iyy, window)

    # Step 2 calculate response
    # determinant
    detA = Axx * Ayy - Axy ** 2
    # trace
    traceA = Axx + Ayy
    # response
    response = detA - k * traceA ** 2

    # step 3. Get points location(x,y)
    points = corner_peaks(response)

    return points


def display_corner_points(org_img:np.ndarray, points, output_name):
    bool_arr = np.zeros(org_img.shape[:2],dtype=np.int8)
    bool_arr = np.bool8(bool_arr)
    
    # pprint(bool_arr)
    for (x,y) in points:
        bool_arr[x,y] = True
    
    # pprint(bool_arr)
    org_img[bool_arr]=[0,0,255]
    cv2.imwrite(output_name, org_img)




if __name__ == "__main__":
    print ("testing get_harris_points")

    images = [
        "../data/campus/sun_abslhphpiejdjmpz.jpg",
        "../data/campus/sun_dmyfiizrhnceheac.jpg",
        "../data/campus/sun_ahprylpgnmgqiyuz.jpg",
        # "../data/desert/sun_afkfpfcdesbszufn.jpg",
        # "../data/campus/sun_bxwzzaswntmwoyml.jpg",
    ]

    alpha = 5000
    k = 0.05
    idx = 0
    for image in images:
        idx +=1
        I = cv2.imread(image)
        points = get_harris_points(I,alpha,k)
        output_name = f"../output/Q1.2_harris_{idx}.jpg"
        display_corner_points(I,points,output_name)

        I = cv.imread(image)
        points = get_random_points(I,alpha)
        output_name = f"../output/Q1.2_random_{idx}.jpg"
        display_corner_points(I,points,output_name)
