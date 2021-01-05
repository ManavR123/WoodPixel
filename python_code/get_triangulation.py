import numpy as np
import cv2 as cv
from collections import defaultdict
from scipy.interpolate import CubicSpline

np.random.seed(42069)


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


def convert_to_contours(points):
    contour = np.zeros((len(points), 1, 2), dtype=np.int32)
    for i, point in enumerate(points):
        contour[i, 0] = np.array([point[0], point[1]])
    return contour


# Read in image with edges, calculate contours, and draw them
im = cv.imread("alan_edges.png", 0)
image_contours = np.ones((im.shape[0], im.shape[1], 3)) * 255
ret, thresh = cv.threshold(im, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# cv.drawContours(image_contours, [c for c in contours if len(c) > 500], -1, (0, 0, 0), 1)

STEP_SIZE = 100
MIN_DIST = 50

r = (0, 0, im.shape[1], im.shape[0])
subdiv = cv.Subdiv2D(r)
points = []
edge_to_curve = {}


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
    prev_point = None
    curve_points = []
    for i in range(int(distance[-1])):
        point = tuple(np.rint(f(i)).astype(int))
        if i % STEP_SIZE == 0:
            dist = -1
            if points:
                min_neighbor = min(
                    points,
                    key=lambda x: np.sqrt(
                        (x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2
                    ),
                )
                dist = np.sqrt(
                    (min_neighbor[0] - point[0]) ** 2
                    + (min_neighbor[1] - point[1]) ** 2
                )
            if points and dist < MIN_DIST:
                prev_point = None
                curve_points = []
                continue
            points.append(point)
            if prev_point:
                edge = (prev_point, point)
                edge_to_curve[edge] = curve_points
                curve_points = []
            prev_point = point
        elif prev_point:
            curve_points.append(point)

num_random_points = 0
while num_random_points < 100:
    num_random_points += 1
    point = (np.random.randint(im.shape[1]), np.random.randint(im.shape[0]))
    dist = -1
    if points:
        min_neighbor = min(
            points,
            key=lambda x: np.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2),
        )
        dist = np.sqrt(
            (min_neighbor[0] - point[0]) ** 2 + (min_neighbor[1] - point[1]) ** 2
        )
    if points and dist < MIN_DIST:
        continue
    points.append(point)

for point in points:
    cv.circle(image_contours, point, 1, (0, 0, 255), 2)
    subdiv.insert(point)

triangleList = subdiv.getTriangleList()
print(f"Number of triangles: {len(triangleList)}")
delaunay_color = (255, 0, 0)

todo = set()
for t in triangleList:
    pt1 = (int(t[0]), int(t[1]))
    pt2 = (int(t[2]), int(t[3]))
    pt3 = (int(t[4]), int(t[5]))
    if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
        for p1, p2 in [(pt1, pt2), (pt2, pt3), (pt1, pt3)]:
            if (p1, p2) in edge_to_curve:
                todo.add((p1, p2))
            elif (p2, p1) in edge_to_curve:
                todo.add((p2, p1))
            else:
                cv.line(image_contours, p1, p2, delaunay_color, 1, cv.LINE_AA, 0)

for p1, p2, in todo:
    contour = edge_to_curve[(p1, p2)]
    if any([all(image_contours[c[1], c[0]] == delaunay_color) for c in contour]):
        cv.line(image_contours, p1, p2, (0, 0, 255), 1, cv.LINE_AA, 0)
    else:
        cv.drawContours(
            image_contours, convert_to_contours(contour), -1, (0, 255, 0), 1,
        )

cv.imshow("Triangulation", image_contours)
cv.imwrite("alan_triangles.png", image_contours)
cv.waitKey(0)
