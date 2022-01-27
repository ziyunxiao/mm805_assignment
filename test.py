import numpy as np
import cv2
from imutils import paths
import matplotlib.pyplot as plt

from assignment_code import knn_match

def test_knn_match():
    print("test_knn_match")
    img_name1 = './data/office/office-00.png'
    img_name2 = './data/office/office-01.png'

    img1 = cv2.imread(img_name1,cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.imread(img_name2,cv2.IMREAD_GRAYSCALE)

    knn_match(img1,img2,'sift')

test_knn_match()    