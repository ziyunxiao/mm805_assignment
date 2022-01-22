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
    points = corner_peaks(response,min_distance=4)
    # points = get_coordinate(response,I.shape)

    return points

def get_coordinate(response, image_shape,alpha=5000):
    # sort with flattern mode
    sortedIndex = np.argsort(response,axis=None)[::-1]

    # maxIndes are index of flattern array
    maxIndexes = sortedIndex[0:alpha]

    # generate points coordinates
    # consider image as coordianate in place m x n
    # each point coordnate is (m', n')
    # m' * n + n' = flatterned index
    m = image_shape[0]
    n = image_shape[1]

    points = np.zeros((2,alpha),dtype=np.int32)
    x = np.floor(maxIndexes/n)
    y = np.mod(maxIndexes,n)
    points[0,:] = x
    points[1,:] = y
    points = points.T

    # pprint(points)

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
    print("main")