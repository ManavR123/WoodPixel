import numpy as np
import cv2 as cv
from scipy.interpolate import CubicSpline

# Read in image with edges, calculate contours, and draw them
im = cv.imread("alan_edges.png", 0)
image_contours = np.ones((im.shape[0], im.shape[1], 3)) * 255
ret, thresh = cv.threshold(im, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(image_contours, [c for c in contours if len(c) > 500], -1, (0, 0, 0), 1)

cv.imwrite("alan_contours.png", image_contours)


def is_extendable(im, point):
    """
    Helper function to determine if we should continue to extend a point.
    This is based on if the point is still within the image boundaries and
    if the pixel corresponding to the point is white enough (this indicates)
    we haven't hitten another edge.
    """
    point = np.rint(point).astype(int)
    return (
        point[0] >= 0
        and point[0] < im.shape[1]
        and point[1] >= 0
        and point[1] < im.shape[0]
        and im[point[1], point[0]] >= 100
    )


# Iterate through all of the contours and for contours that have enough points
# calculate a spline and step through the function and every so often calculate
# the normal to the point and step in both direction drawing a line until you hit
# another edge.
for i in range(0, len(contours)):
    # only look at contours with at least 500 points
    if len(contours[i]) < 500:
        continue
    print(contours[i].shape)
    # need to remove extra dummy dimension
    c = contours[i].squeeze(1)

    # Calculate distances to parameterize the contour and create a spline
    # from distance to pixel location.
    distance = np.cumsum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    f = CubicSpline(distance, c)

    # Step through points in the spline
    for i in range(0, int(distance[-1]), 1):
        # uncomment line below to see various points on the spline
        # cv.circle(image_contours, tuple(np.rint(f(i)).astype(int)), 1, (0,0,0), 1)

        # every 70 steps take the normal of the point
        if i % 70 == 0:
            derivative = f(i, 1)
            normal = np.array([-derivative[1], derivative[0]])

            # move in the (opposite) direction of the normal to the point
            # until you hit another edge
            ep_values = np.array([-1, 1]) * 0.5
            for eps in ep_values:
                point = f(i)
                while is_extendable(im, point):
                    point += normal * eps

                # draw line from starting point to ending point
                cv.line(
                    image_contours,
                    tuple(np.rint(f(i)).astype(int)),
                    tuple(np.rint(point).astype(int)),
                    (0, 0, 0),
                    1,
                )

cv.imwrite("alan_contours_segmented.png", image_contours)
