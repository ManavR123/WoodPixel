import argparse
import json
import os
import shutil
from collections import defaultdict

import cv2 as cv
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from TriangleRegion import TriangleRegion
from wn_test import wn_PnPoly

np.random.seed(42069)
# Constants
STEP_SIZE = 100
MIN_DIST = 50
EPS = 0.5
EDGE_TOLERANCE = 10
COUNTOUR_THRESHOLD = 500
NUM_RANDOM_ATTEMPTS_THRESHOLD = 1000
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
    """
    Helper function to create a spline function from a set of contour points
    """
    # Calculate distances to parameterize the points and create a spline
    # from distance to pixel location.
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    f = CubicSpline(distance, points)
    return f, distance


def invalid_point(point, points, img):
    min_neighbor = min(
        points,
        key=lambda x: np.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2),
    )
    dist = np.sqrt((min_neighbor[0] - point[0]) ** 2 + (min_neighbor[1] - point[1]) ** 2)
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


def invalid_curve_triangle(triangles, p1, p2, f, distance, img, threshold=0.1, eps=0.1):
    """
    Checks if a contour given as a spline function crosses any of the edges in the quadrilateral
    it is bounded by. threshold determines a tolerance of how close it may be and eps is the step
    size by which we walk along the contour.
    """
    assert len(triangles) == 2, f"There should only be 2 triangles sharing an edge, you had {len(triangles)}"
    edges = [edge for t in triangles for edge in t.edge_map if edge != (p1, p2) and edge != (p2, p1)]
    assert len(edges) == 4, f"Inconsistent number of edges, {len(edges)}"
    for i in np.arange(0, distance[-1], eps):
        point = f(i)
        # uncomment line below to visualize steps
        # cv.circle(img, tuple(np.rint(point).astype(int)), 1, (0, 0, 255), 0)
        # if point is threshold away from each edge or within eps of a vertex, it a valid else its invalid
        if (
            any([projection_dist(np.array(edge[0]), np.array(edge[1]), np.array(point)) < threshold for edge in edges])
        ):
            return True
    return False


def projection_dist(a, b, p):
    """
    Calculate thr distance from point p to the line ab.
    @source - https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
    """
    return np.linalg.norm(np.cross(b - a, a - p)) / np.linalg.norm(b - a)


def get_border_points(im):
    """Add points around the border of the image"""
    points = []
    points += [(0, i) for i in range(0, im.shape[0] - STEP_SIZE, STEP_SIZE)]
    points.append((0, im.shape[0] - 1))

    points += [(im.shape[1] - 1, i) for i in range(0, im.shape[0] - STEP_SIZE, STEP_SIZE)]
    points.append((im.shape[1] - 1, im.shape[0] - 1))

    points += [(i, 0) for i in range(0, im.shape[1] - STEP_SIZE, STEP_SIZE)]
    points.append((im.shape[1] - 1, 0))

    points += [(i, im.shape[0] - 1) for i in range(0, im.shape[1] - STEP_SIZE, STEP_SIZE)]
    points.append((im.shape[1] - 1, im.shape[0] - 1))
    return points


def main(inputfile, outputdir):
    # Read in image with edges and calculate contours
    im = cv.imread(inputfile, 0)
    image_contours = np.ones((im.shape[0], im.shape[1], 3)) * 255
    ret, thresh = cv.threshold(im, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Keep of a list of points to use for the triangles
    points = []
    points += get_border_points(im)

    # Traverse the contours of the image and samples points every STEP_SIZE
    # such that they are MIN_DIST away from existing points. Store points in between
    # sampled points and store them in edge_to_curve.
    edge_to_curve = {}
    for i in range(0, len(contours)):
        # only look at contours with at least COUNTOUR_THRESHOLD points
        if len(contours[i]) < COUNTOUR_THRESHOLD:
            continue

        # uncomment line below to visualize contours
        # cv.drawContours(image_contours, contours, i, (255, 0, 0), 1)

        # need to remove extra dummy dimension
        c = contours[i].squeeze(1)
        f, distance = make_spline(c)

        # Step through points in the spline by EPS steps. Store the last control point
        # and a list of points we stepped through since the last control point
        last_ctrl_point, prev_point = None, None
        curve_points = []
        for i in np.arange(0, int(distance[-1]), EPS):
            point = tuple(np.rint(f(i)).astype(int))
            if i % STEP_SIZE == 0:
                # Check if point is valid
                if points and invalid_point(point, points, image_contours):
                    last_ctrl_point = None
                    curve_points = []
                    continue
                points.append(point)
                if last_ctrl_point:
                    edge = (last_ctrl_point[0], point)
                    edge_to_curve[edge] = curve_points
                    curve_points = []
                last_ctrl_point = (point, i)
                prev_point = point
            elif last_ctrl_point and point != prev_point:
                prev_point = point
                curve_points.append(point)

    # Add up to NUM_RANDOM_POINTS_THRESHOLD random poins across the image
    num_random_points, num_attempts = 0, 0
    while num_random_points < NUM_RANDOM_POINTS_THRESHOLD and num_attempts < NUM_RANDOM_ATTEMPTS_THRESHOLD:
        print(f"{num_random_points}/{num_attempts}", end="\r")
        num_attempts += 1
        point = (
            np.rint(np.random.normal(im.shape[1] / 2, 75)).astype(int),
            np.rint(np.random.normal(im.shape[0] / 2, 75)).astype(int),
        )
        if points and invalid_point(point, points, image_contours):
            continue
        points.append(point)
        num_random_points += 1
    print(f"{num_random_points}/{num_attempts} random points added")

    # Create a rectangle that bound the image so it can be subdivided with
    # Delaunay triangles
    r = (0, 0, im.shape[1], im.shape[0])
    subdiv = cv.Subdiv2D(r)
    for point in points:
        # uncomment line below to visualize chosen points
        # cv.circle(image_contours, point, 1, (0, 0, 255), 2)
        subdiv.insert(point)

    triangleList = subdiv.getTriangleList()
    print(f"Number of triangles created: {len(triangleList)}")

    # Set to different colors to visualize different edges
    EDGE_COLOR = (0, 0, 0)
    FIXED_CURVE_COLOR = (0, 255, 0)
    CURVE_COLOR = (255, 0, 0)

    # Iterate through triangles and create TriangleRegion objects and draw lines connecting. For edges that contain points
    # that are both on a contour, determine if the contour between the two points would function as a valid border
    # if it does, use that as the edge, else just use the line.
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
        f, distance = make_spline(contour)
        if invalid_curve_triangle(triangles, p1, p2, f, distance, image_contours):
            cv.line(image_contours, p1, p2, FIXED_CURVE_COLOR, 1, cv.LINE_AA, 0)
        else:
            for t in triangles:
                t.set_contour(p1, p2, contour)
            cv.drawContours(
                image_contours,
                convert_to_contours(contour),
                -1,
                CURVE_COLOR,
                1,
            )

    # Create output json
    all_pixels = []
    bad_pixels = []
    pixel_to_triangle = {}
    output = np.zeros((im.shape[0], im.shape[1], 3))
    print("Creating outputs...")
    if os.path.isdir(outputdir):
        shutil.rmtree(outputdir)
    os.mkdir(outputdir)
    patches_dir = os.path.join(outputdir, "patches")
    os.mkdir(patches_dir)
    patches = []
    for t in tqdm(triangle_regions):
        # form bounding box for the patch
        sorted_edges = [(t.p1, t.p2), (t.p2, t.p3), (t.p3, t.p1)]
        edge_pixels = []
        for edge in sorted_edges:
            edge_pixels.append(edge[0])
            if t.edge_map[edge]:
                edge_pixels.extend(t.edge_map[edge])
        if not len(edge_pixels) == len(set(edge_pixels)):
            print(f"Edge pixels contains duplicates. There are {len(edge_pixels) - len(set(edge_pixels))} duplicates. These are all of the points {edge_pixels}")
        x, y, w, h = cv.boundingRect(np.expand_dims(np.array(edge_pixels), axis=1))
    
        # create binary mask that is only one for pixels
        # that are actually in the patch
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                point = (x + i, y + j)
                if wn_PnPoly(point, edge_pixels):
                    if point in pixel_to_triangle:
                        other_t = pixel_to_triangle[point]
                        print(f"Triangle {t.id} tried to claim {point} but Triangle {other_t.id} already has it")
                    else:
                        pixel_to_triangle[point] = t
                    mask[j, i] = 255
                    if all(output[point[1], point[0]] == (255, 255, 255)) or all(output[point[1], point[0]] == (0, 0, 255)):
                        bad_pixels.append(point)
                        output[point[1], point[0]] = (0, 0, 255)
                    else:
                        output[point[1], point[0]] = (255, 255, 255)
                    all_pixels.append(point)

        mask_path = os.path.join(patches_dir, f"{t.id:08}.bmp")
        cv.imwrite(mask_path, mask)

        centroid = ((t.p1[0] + t.p2[0] + t.p3[0]) // 3, (t.p1[1] + t.p2[1] + t.p3[1]) // 3)
        cv.putText(image_contours, f"{t.id}", centroid, cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
 
        # Create patch entry
        entry = {
            "curves_top": "",
            "curves_bot": "",
            "curves_left": "",
            "curves_right": "",
            "curves_diag": "",
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "mask": os.path.join("patches", f"{t.id:08}.bmp"),
            "sub_regions": "",
            "rectangular": "false",
            "target_index": "0",
            "coordinate": {"x": "0", "y": "0"},
        }

        for i, edge in enumerate(sorted_edges, 1):
            entry[f"curve_{i}"] = {"knot_points": [{"x": int(kp[0]), "y": int(kp[1])} for kp in t.get_knots(*edge)]}

        patches.append(entry)

    with open(os.path.join(outputdir, "grid.json"), "w") as f:
        f.write(json.dumps({"patches": patches}, indent=4))
    cv.imwrite(os.path.join(outputdir, "triangles.png"), image_contours)

    assert len(all_pixels) == (im.shape[0] - 1) * (im.shape[1] - 1), f"Missed a few pixels. Expected {(im.shape[0] - 1) * (im.shape[1] - 1)}. Got {len(all_pixels)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to take an image of contours and outputs a directory of the patch masks and a json to input into fit_patches."
    )
    parser.add_argument("--input", type=str, help="path to input file")
    parser.add_argument("--output", type=str, help="path to output directory")
    args = parser.parse_args()

    assert args.input and args.output

    main(args.input, args.output)
