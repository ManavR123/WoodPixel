import numpy as np
import cv2 as cv
from collections import defaultdict
from scipy.interpolate import CubicSpline
import argparse

np.random.seed(42069)
# Constants
STEP_SIZE = 100
MIN_DIST = 50
EPS = 0.5
EDGE_TOLERANCE = 10


def rect_contains(rect, point):
    """
    Helper function to check if a point is bounded by
    a rectangle.
    """
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
    """
    Convert a list of points into the proper shape of a contour.
    """
    contour = np.zeros((len(points), 1, 2), dtype=np.int32)
    for i, point in enumerate(points):
        contour[i, 0] = np.array([point[0], point[1]])
    return contour


def invalid_point(point, dist, img):
    return dist < MIN_DIST or point[0] < EDGE_TOLERANCE or point[1] < EDGE_TOLERANCE or point[0] >= img.shape[0] or point[1] >= img.shape[1]

def invalid_curve(points, p1, p2, img, edge_color):
    """
    Check if a list of points cross an existing edge in the image.
    """

    THRESHOLD = 1
    has_taken_vacation = False
    length_of_break = 0
    for c in points:
        if c == p1 or c == p2:
            continue
        if any(img[c[1], c[0]] != edge_color):
            if length_of_break > THRESHOLD and has_taken_vacation:
                return True  # (we already reached an edge, but now we left it again!? bad)
            length_of_break += 1
        else:
            if length_of_break > THRESHOLD:
                has_taken_vacation = True
            length_of_break = 0  # (we left an edge earlier, but now we're back on an edge, so set arrival flag)
    return False

def main(inputfile, outputfile):
    # Read in image with edges, calculate contours, and optionally draw them
    im = cv.imread(inputfile, 0)
    image_contours = np.ones((im.shape[0], im.shape[1], 3)) * 255
    ret, thresh = cv.threshold(im, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(image_contours, [c for c in contours if len(c) > 500], -1, (0, 0, 0), 1)

    # Keep of a list of points to use for the triangles
    points = []
    # Keep a map of points that are in between consecutive points on the image contours
    edge_to_curve = {}

    # Add points around the border of the image
    points += [(0, i) for i in range(0, im.shape[0] - STEP_SIZE, STEP_SIZE)]
    points.append((0, im.shape[0] - 1))

    points += [(im.shape[1] - 1, i) for i in range(0, im.shape[0] - STEP_SIZE, STEP_SIZE)]
    points.append((im.shape[1] - 1, im.shape[0] - 1))

    points += [(i, 0) for i in range(0, im.shape[1] - STEP_SIZE, STEP_SIZE)]
    points.append((im.shape[1] - 1, 0))

    points += [(i, im.shape[0] - 1) for i in range(0, im.shape[1] - STEP_SIZE, STEP_SIZE)]
    points.append((im.shape[1] - 1, im.shape[0] - 1))

    # Traverse the contours of the image and samples points every STEP_SIZE
    # such that they are MIN_DIST away from existing points. Store points in between
    # sampled points and store them in edge_to_curve.
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

        # Step through points in the spline by EPS steps. Store the last control point
        # and a list of points we stepped through since the last control point
        prev_point = None
        curve_points = []
        for i in np.arange(0, int(distance[-1]), EPS):
            point = tuple(np.rint(f(i)).astype(int))
            if i % STEP_SIZE == 0:
                # Enforce MIN_DIST constraint
                dist = -1
                if points:
                    min_neighbor = min(points, key=lambda x: np.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2))
                    dist = np.sqrt((min_neighbor[0] - point[0]) ** 2 + (min_neighbor[1] - point[1]) ** 2)
                if points and invalid_point(point, dist, image_contours):
                    prev_point = None
                    curve_points = []
                    continue
                points.append(point)
                if prev_point:
                    edge = (prev_point, point)
                    edge_to_curve[edge] = curve_points + [point]
                    curve_points = [point]
                prev_point = point
            elif prev_point:
                curve_points.append(point)

    # Add up to 100 random poins across the image
    num_random_points, num_attempts = 0, 0
    while num_random_points < 100 and num_attempts < 1000:
        print(num_attempts, num_random_points)
        num_attempts += 1
        point = (np.rint(np.random.normal(im.shape[1] / 2, 75)).astype(int), np.rint(np.random.normal(im.shape[0] / 2, 75)).astype(int))
        dist = -1
        if points:
            min_neighbor = min(points, key=lambda x: np.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2),)
            dist = np.sqrt((min_neighbor[0] - point[0]) ** 2 + (min_neighbor[1] - point[1]) ** 2)
        if points and invalid_point(point, dist, image_contours):
            continue
        points.append(point)
        num_random_points += 1

    # Create a rectangle that bound the image so it can be subdivided with
    # Delaunay triangles
    r = (0, 0, im.shape[1], im.shape[0])
    subdiv = cv.Subdiv2D(r)
    # Insert all of our points into the sub-division
    for point in points:
        # cv.circle(image_contours, point, 1, (0, 0, 255), 2)
        subdiv.insert(point)

    triangleList = subdiv.getTriangleList()
    print(f"Number of triangles: {len(triangleList)}")

    # Set to different colors to visualize different edges
    EDGE_COLOR = (0, 0, 0)
    FIXED_CURVE_COLOR = (0, 0, 0)
    CURVE_COLOR = (0, 0, 0)

    # Iterate through tri
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
                    cv.line(image_contours, p1, p2, EDGE_COLOR, 1, cv.LINE_AA, 0)

    for p1, p2, in todo:
        contour = edge_to_curve[(p1, p2)]
        if invalid_curve(contour, p1, p2, image_contours, EDGE_COLOR):
            cv.line(image_contours, p1, p2, FIXED_CURVE_COLOR, 1, cv.LINE_AA, 0)
        else:
            cv.drawContours(
                image_contours, convert_to_contours(contour), -1, CURVE_COLOR, 1,
            )

    cv.imwrite(outputfile, image_contours)
    cv.imshow("Output", image_contours)
    cv.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to take an image segmented into regions and outputs a directory of the patch masks and a json to input into fit_patches."
    )
    parser.add_argument("--input", type=str, help="path to input file")
    parser.add_argument("--output", type=str, help="path to output directory")
    args = parser.parse_args()

    assert args.input and args.output

    main(args.input, args.output)