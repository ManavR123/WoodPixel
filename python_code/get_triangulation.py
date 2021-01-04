import numpy as np
import cv2 as cv
from scipy.interpolate import CubicSpline

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True



# Read in image with edges, calculate contours, and draw them
im = cv.imread("alan_edges.png", 0)
image_contours = np.ones((im.shape[0], im.shape[1], 3)) * 255
ret, thresh = cv.threshold(im, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# cv.drawContours(image_contours, [c for c in contours if len(c) > 500], -1, (0, 0, 0), 1)
STEP_SIZE = 50
MIN_DIST = 50

r = (0, 0, im.shape[1], im.shape[0])
subdiv  = cv.Subdiv2D(r)
points = []

points += [(0, i) for i in range(0, im.shape[0], STEP_SIZE)]
points += [(im.shape[1] - 1, i) for i in range(0, im.shape[0], STEP_SIZE)]
points += [(i, 0) for i in range(0, im.shape[1], STEP_SIZE)]
points += [(i, im.shape[0] - 1) for i in range(0, im.shape[1], STEP_SIZE)]

for i in range(0, len(contours)):
    # only look at contours with at least 500 points
    if len(contours[i]) < 500:
        continue

    # need to remove extra dummy dimension
    c = contours[i].squeeze(1)

    # Calculate distances to parameterize the contour and create a spline
    # from distance to pixel location.
    distance = np.cumsum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    f = CubicSpline(distance, c)

    # Step through points in the spline
    for i in range(0, int(distance[-1]), STEP_SIZE):
        # uncomment line below to see various points on the spline
        point = tuple(np.rint(f(i)).astype(int))
        dist = -1
        if points:
            min_neighbor = min(points, key=lambda x: np.sqrt((x[0] - point[0])**2 + (x[1] - point[1])**2))
            dist = np.sqrt((min_neighbor[0] - point[0])**2 + (min_neighbor[1] - point[1])**2)
        if points and dist < MIN_DIST:
            continue
        points.append(point)

num_random_points = 0
while num_random_points < 100:
    num_random_points += 1
    point = (np.random.randint(im.shape[1]), np.random.randint(im.shape[0]))
    dist = -1
    if points:
        min_neighbor = min(points, key=lambda x: np.sqrt((x[0] - point[0])**2 + (x[1] - point[1])**2))
        dist = np.sqrt((min_neighbor[0] - point[0])**2 + (min_neighbor[1] - point[1])**2)
    if points and dist < MIN_DIST:
        continue
    points.append(point)

for point in points:
    cv.circle(image_contours, point, 1, (0,0,255), 2)
    subdiv.insert(point)

triangleList = subdiv.getTriangleList()
print(len(triangleList))
delaunay_color = (255, 0, 0)
for t in triangleList:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
        cv.line(image_contours, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
        cv.line(image_contours, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
        cv.line(image_contours, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)

cv.imshow("Triangulation", image_contours)
cv.waitKey(0)
