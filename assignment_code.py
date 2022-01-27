import numpy as np
import cv2
from pprint import pprint
from scipy import ndimage
from skimage.feature import corner_peaks
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


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

def BF_match_orb(img1, img2):
    ''' based on open_cv example code '''    
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(img3)
    plt.title('orb')
    plt.show()
   

def BF_match_sift(img1,img2):
    '''Based on OpenCV example code'''
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(img3)
    plt.title('sift')
    plt.show()

def knn_match(img1,img2,descriptor='sift',show_limit=100):
    '''Only support sift and orb as descriptor'''

    # Initiate SIFT detector
    if descriptor == 'sift':
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        # descriptors are 128 dimention histogram,
        # to compare descriptors using Knn 
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    elif descriptor == 'orb':
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
    else:
        print(f"Unsupported descriptor {descriptor}")
        return

    # KNN match 
    k = 2
    m = des1.shape[0]
    n = des2.shape[0]
    
    matches = list()
    dis_list = pairwise_distances(des1,des2,metric='euclidean')
    for i in range(m):
        d_list = list()
        arr = dis_list[i,:]
        arr = np.argsort(arr,axis=0)[0:k]
        for j in range(k):
            idx_j = arr[j]
            d = cv2.DMatch()
            d.distance = dis_list[i,idx_j]
            d.imgIdx = 0
            d.queryIdx = i
            d.trainIdx = idx_j
            d_list.append(d)

        # add to matches
        matches.append(d_list)
    
    matches = np.array(matches)
    r1 = matches.shape[0]

    
    good = []
    dis_values = list()    
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            dis_values.append(m.distance)
    
    r2 = len(good)
    print(f"Descriptor {descriptor} detects {r1} points and {r2} points are good")

    # sort good 
    sorted_idx = np.argsort(dis_values)[:show_limit]
    good_show = []
    for idx in sorted_idx:
        good_show.append(good[idx])
    
    # Display result on plot
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_show,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(img3)
    plt.title(descriptor)
    plt.show()



if __name__ == "__main__":
    print("main")