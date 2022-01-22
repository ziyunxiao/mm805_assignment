import assignment_code as ac
import numpy as np
from pprint import pprint
import cv2
from skimage.feature import corner_harris, corner_peaks, corner_subpix

def test1():
    I = np.zeros((5, 5))
    I[2:4, 2:4] = 1
    points = ac.get_harris_points(I)
    pprint(points)


def test2():
    print ("testing get_harris_points")

    images = [
        "data/carmel/carmel-00.png",
        "data/carmel/carmel-01.png",
        "data/office/office-00.png",
        "data/office/office-01.png",
    ]

    idx = 0
    for image in images:
        idx +=1
        I = cv2.imread(image)
        grey = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        grey = grey / 255.0

        points = ac.get_harris_points(grey)
        output_name = f"./output/test_harris_{idx}.jpg"
        ac.display_corner_points(I,points,output_name)

        points2 = corner_peaks(corner_harris(grey), min_distance=4)
        output_name = f"./output/test2_harris_{idx}.jpg"
        ac.display_corner_points(I,points2,output_name)

        print(f" {points.shape} - {points2.shape}")

# test
test2()