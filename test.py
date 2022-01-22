import assignment_code as ac
import numpy as np
from pprint import pprint

def test1():
    I = np.zeros((5, 5))
    I[2:4, 2:4] = 1
    points = ac.get_harris_points(I)
    pprint(points)

test1()

