import numpy as np
import cv2 as cv
from collections import defaultdict
from scipy.interpolate import CubicSpline
from TriangleRegion import TriangleRegion
import argparse

np.random.seed(42069)
# Constants
STEP_SIZE = 100
MIN_DIST = 50
EPS = 0.5
EDGE_TOLERANCE = 10
COUNTOUR_THRESHOLD = 500
NUM_ATTEMPTS_THRESHOLD = 1000
NUM_RANDOM_POINTS_THRESHOLD = 100


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


def make_spline(points):
    # assert len(points) == len(set([tuple(p) for p in points])), "Points contains duplicates"
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    f = CubicSpline(distance, points)
    return f, distance


def invalid_point(point, dist, img):
    return (
        dist < MIN_DIST
        or point[0] < EDGE_TOLERANCE
        or point[1] < EDGE_TOLERANCE
        or point[0] >= img.shape[0]
        or point[1] >= img.shape[1]
    )


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


def invalid_curve_triangle(triangles, p1, p2, f, distance, img, threshold=0.01, eps=0.5):
    assert len(triangles) == 2, f"There should only be 2 triangles sharing an edge, you had {len(triangles)}"
    edges = [edge for t in triangles for edge in t.edge_map if edge != (p1, p2) and edge != (p2, p1)]
    assert len(edges) == 4, f"Too many edges, {len(edges)}"
    for i in np.arange(0, distance[-1], eps):
        point = f(i)
        if (
            any([projection_dist(np.array(edge[0]), np.array(edge[1]), np.array(point)) < threshold for edge in edges])
            and np.linalg.norm(point - np.array(list(p2))) > eps and np.linalg.norm(point - np.array(list(p1))) > eps
        ):
            cv.circle(img, tuple(np.rint(point).astype(int)), 1, (0, 0, 255), 1)
            return True
    print("False")
    return False


def projection_dist(a, b, p):
    ret = np.linalg.norm(np.cross(b - a, a - p)) / np.linalg.norm(b - a)
    return ret


def main(inputfile, outputfile):
    # Read in image with edges, calculate contours, and optionally draw them
    im = cv.imread(inputfile, 0)
    image_contours = np.ones((im.shape[0], im.shape[1], 3)) * 255
    ret, thresh = cv.threshold(im, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(image_contours, [c for c in contours if len(c) > COUNTOUR_THRESHOLD], -1, (255, 0, 0), 1)

    # Keep of a list of points to use for the triangles
    points = []
    # Keep a map of points that are in between consecutive points on the image contours
    edge_to_curve = {}
    edge_to_spline = {}

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
        # only look at contours with at least COUNTOUR_THRESHOLD points
        if len(contours[i]) < COUNTOUR_THRESHOLD:
            continue

        # need to remove extra dummy dimension
        c = contours[i].squeeze(1)

        # Calculate distances to parameterize the contour and create a spline
        # from distance to pixel location.
        f, distance = make_spline(c)

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
                    min_neighbor = min(
                        points,
                        key=lambda x: np.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2),
                    )
                    dist = np.sqrt((min_neighbor[0] - point[0]) ** 2 + (min_neighbor[1] - point[1]) ** 2)
                if points and invalid_point(point, dist, image_contours):
                    prev_point = None
                    curve_points = []
                    continue
                points.append(point)
                if prev_point:
                    edge = (prev_point[0], point)
                    edge_to_curve[edge] = curve_points
                    curve_points = []
                    edge_to_spline[edge] = (f, prev_point[1], i)
                prev_point = (point, i)
            elif prev_point:
                curve_points.append(point)

    # Add up to NUM_RANDOM_POINTS_THRESHOLD random poins across the image
    num_random_points, num_attempts = 0, 0
    while num_random_points < NUM_RANDOM_POINTS_THRESHOLD and num_attempts < NUM_ATTEMPTS_THRESHOLD:
        print(num_attempts, num_random_points)
        num_attempts += 1
        point = (
            np.rint(np.random.normal(im.shape[1] / 2, 75)).astype(int),
            np.rint(np.random.normal(im.shape[0] / 2, 75)).astype(int),
        )
        dist = -1
        if points:
            min_neighbor = min(
                points,
                key=lambda x: np.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2),
            )
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
    FIXED_CURVE_COLOR = (0, 255, 0)
    CURVE_COLOR = (255, 0, 0)

    # Iterate through tri
    triangle_regions = []
    todo = set()
    edge_to_triangle = defaultdict(list)
    id = 0
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        t = TriangleRegion(pt1, pt2, pt3, id)
        triangle_regions.append(t)
        id += 1
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            for p1, p2 in [(pt1, pt2), (pt2, pt3), (pt1, pt3)]:
                edge_to_triangle[(p1, p2)].append(t)
                edge_to_triangle[(p2, p1)].append(t)
                if (p1, p2) in edge_to_curve:
                    todo.add((p1, p2))
                elif (p2, p1) in edge_to_curve:
                    todo.add((p2, p1))
                else:
                    cv.line(image_contours, p1, p2, EDGE_COLOR, 1, cv.LINE_AA, 0)

    for (p1, p2) in todo:
        contour = edge_to_curve[(p1, p2)]
        triangles = edge_to_triangle[(p1, p2) if (p1, p2) in edge_to_triangle else (p2, p1)]
        f, distance = make_spline(list(set(contour)))
        if invalid_curve_triangle(triangles, p1, p2, f, distance, image_contours):
            cv.line(image_contours, p1, p2, FIXED_CURVE_COLOR, 1, cv.LINE_AA, 0)
        if True:
            [t.set_contour(p1, p2, contour) for t in triangles]
            cv.drawContours(
                image_contours,
                convert_to_contours(contour),
                -1,
                CURVE_COLOR,
                1,
            )
            global_f, t1, t2 = edge_to_spline[(p1, p2)]
            global_contour = [global_f(i) for i in np.arange(t1, t2, EPS)]
            # cv.drawContours(
            #     image_contours,
            #     convert_to_contours(global_contour),
            #     -1,
            #     (255, 0, 255),
            #     1,
            # )

    cv.imwrite(outputfile, image_contours)
    # cv.imshow("Output", image_contours)
    cv.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to take an image of contours and outputs a directory of the patch masks and a json to input into fit_patches."
    )
    parser.add_argument("--input", type=str, help="path to input file")
    parser.add_argument("--output", type=str, help="path to output directory")
    args = parser.parse_args()

    assert args.input and args.output

    main(args.input, args.output)
