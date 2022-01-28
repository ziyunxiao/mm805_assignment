import numpy as np
import cv2
from imutils import paths
import matplotlib.pyplot as plt
from assignment_code import knn_match, feature_match_by_tracking, LK_OpticalFlow
from pprint import pprint


def test_knn_match():
    print("test_knn_match")
    img_name1 = './data/office/office-00.png'
    img_name2 = './data/office/office-01.png'

    img1 = cv2.imread(img_name1,cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.imread(img_name2,cv2.IMREAD_GRAYSCALE)

    knn_match(img1,img2,'sift')

def test_lk_optic_flow(image_folder):
    # image_folder = "./data/cave_3"
    imagePaths = sorted(list(paths.list_images(image_folder)))
    images = []

    # images to stitch list
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)

    img1 = images[0]
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = images[1]
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    u,v = LK_OpticalFlow(img1,img2)
    pprint(u)
    pprint(v)

# test_knn_match()    
image_folder = './data/cave_3'
# feature_match_by_tracking(image_folder,50,10)
test_lk_optic_flow(image_folder)
