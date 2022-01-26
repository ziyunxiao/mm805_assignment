import numpy as np
import cv2
from imutils import paths
import matplotlib.pyplot as plt

from assignment_code import knn_match_sift

def test_knn_match():
    print("test_knn_match")
    img_name1 = './data/office/office-00.png'
    img_name2 = './data/office/office-01.png'

    img1 = cv2.imread(img_name1,cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.imread(img_name2,cv2.IMREAD_GRAYSCALE)

    knn_match_sift(img1,img2)

test_knn_match()    