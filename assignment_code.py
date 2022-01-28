import numpy as np
import cv2
from pprint import pprint
from scipy import ndimage
from skimage.feature import corner_peaks
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from imutils import paths
from scipy import signal

# 1
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


# 1c
def get_pre_points(pre_gray,n):
    pre_points = cv2.goodFeaturesToTrack(pre_gray,n,0.01,10)
    pre_points = np.array(pre_points,dtype='float32')
    kp1 = list()
    for j in range(n):
        d = cv2.KeyPoint()
        d.pt = pre_points[j][0]
        d.size = 31.0
        kp1.append(d)
    return(pre_points,kp1)

def get_new_points(pre_gray,new_gray,pre_points):
    new_points, status, err = cv2.calcOpticalFlowPyrLK(pre_gray,
    new_gray,pre_points,None,maxLevel=2)

    n = pre_points.shape[0]
    # show matching
    # convert format to keypoints
    kp2 = list()
    matches = list()
    for j in range(n):
        good = status[j,0]

        d = cv2.KeyPoint()
        d.pt = new_points[j,0]
        d.size = 31
        kp2.append(d)

        if good != 0:
            m = cv2.DMatch()
            m.queryIdx = j
            m.trainIdx = j
            matches.append(m)

    return (new_points,kp2,matches)

def feature_match_by_tracking(image_folder,points_show=50,min_threshold=40):
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(image_folder)))
    images = []

	# images to stitch list
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)

    # it can be chagned to read mp4 if needed    
    n = 100
    pre_img  = images[0]
    pre_gray = cv2.cvtColor(pre_img,cv2.COLOR_BGR2GRAY)
    (pre_points, kp1) = get_pre_points(pre_gray,n)
    
    for i in range(1,len(images)):
        new_img  = images[i]
        new_gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)

        (new_points,kp2,matches) = get_new_points(pre_gray,new_gray,pre_points)
        m = len(matches)
        
        print(f"frame {i}: good points {m}")

        if m < min_threshold:
            # todo recalculate points
            # recalulate pre_points
            print(f"good points is lower than {min_threshold}, recalculating...")
            (pre_points, kp1) = get_pre_points(pre_gray,n)
            (new_points,kp2,matches) = get_new_points(pre_gray,new_gray,pre_points)


        img3 = cv2.drawMatches(pre_img,kp1,new_img,kp2,\
            matches[:points_show], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(15, 10))
        plt.imshow(img3)
        plt.title('feature match by tracking')
        plt.show()

        pre_img = new_img.copy()
        pre_points = new_points.copy()
        pre_gray = new_gray.copy()
        kp1 = kp2.copy()



#2a Lucas-Kanade optical flow, return u, v (dx,dy)
def LK_OpticalFlow(gray1,gray2,win_size=(3,3)):
    '''
    This function implements the LK optical flow without pyramid or consider as level 1 pyramid.
    '''

    # 1. apply Guassian filter to smooth the image in order to remove noise
    I1 = cv2.GaussianBlur(gray1, win_size, 0)
    I2 = cv2.GaussianBlur(gray2, win_size, 0)
    m,n = win_size
    m = m//2
    n = n//2
    
    # Calculate Ix, Iy, It
    # The filter multiple 1/4 for average result
    filter_x = np.array([[-1,1],[-1,1]]) * 0.25
    filter_y = np.array([[-1,-1],[1,1]]) * 0.25
    filter_t1 = np.array([[1,1],[1,1]]) * 0.25
    filter_t2 = np.array([[1,1],[1,1]]) * -1 * 0.25

    Ix = signal.convolve2d(I1,filter_x,'same') + signal.convolve2d(I2,filter_x,'same')
    Iy = signal.convolve2d(I1,filter_y,'same') + signal.convolve2d(I2,filter_y,'same')
    It = signal.convolve2d(I1,filter_t1,'same') + signal.convolve2d(I2,filter_t2,'same')

    # retrieve good features to track
    points = cv2.goodFeaturesToTrack(I1,1000, 0.01,10)
    points = np.int0(points)

    #init u, v
    u = np.zeros(gray1.shape)
    v = np.zeros(gray1.shape)

    # todo: 
    # Calculating the u and v arrays for the good features obtained n the previous step.
    kp = list()
    status = list()
    err = list()
    for p in points:
        y,x = p.ravel()

        # calculating the derivatives for the neighbouring pixels
        # windows size mxn        

        IX = Ix[x-m:x+m+1,y-n:y+n+1].flatten()
        IY = Iy[x-m:x+m+1,y-n:y+n+1].flatten()
        IT = It[x-m:x+m+1,y-n:y+n+1].flatten()

        # Resolve minimum least squares equation
        # x = inverse(At dot A) dot (At dot b)

        # np.concatenate((IX,IY),axis=1)
        At = np.matrix((IX,IY))
        A = np.matrix.transpose(At)
        A = np.array(A)
        At = np.array(At)
        b = np.transpose(np.array(IT))

        At_dot_A = np.dot(At,A)
        A_inv = np.linalg.pinv(At_dot_A)
        B = np.dot(At,b)

        X = np.dot(A_inv,B)
        u[x,y] = X[0]
        v[x,y] = X[1]
        if X[0] != 0 or X[1] != 0:
            status.append(1)
        else:
            status.append(0)
        kp.append(p)
        kp.append(0)

        # print(f"({x,y}): {X[0], X[1]}")

    return (u, v, points, status, err)

# 2b Visualize result
def display_opitical_flow(image_folder, show_n=10, threshold = 0.5):
    imagePaths = sorted(list(paths.list_images(image_folder)))
    images = []

    # images to stitch list
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)

    img1 = images[0]
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    n = min(len(images),show_n)+1
    for i in range(1,n):
        img2 = images[i]
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        (u, v, points, status, err) = LK_OpticalFlow(gray1,gray2, win_size=(3,3))
        
        # display result
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(400,300))
        plt.subplot(1,3,3)
        plt.title('Vector plot of Optical Flow of good features')
        plt.imshow(img)
        for i in range(len(points)):
            p = points[i]
            y,x = p.ravel()
            # only show points value > threshold
            if status[i] == 1 and(abs(u[x,y])>threshold or abs(v[x,y])>threshold):
                plt.arrow(y,x, v[x,y], u[x,y], width=2,head_width = 5, head_length = 5, color = 'y')
                    
        plt.show()

        img1 = img1
        gray1 = gray1



#3


        



if __name__ == "__main__":
    print("main")