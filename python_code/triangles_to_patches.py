import numpy as np
import cv2 as cv
from collections import defaultdict
from scipy.interpolate import CubicSpline
from PatchRegion import PatchRegion
from tqdm import tqdm

im = cv.imread("tree_triangles.png", 0)
output = np.ones(im.shape, dtype=np.uint8) * 255

print("Calculating PatchRegions...")
edge_map = defaultdict(set)
FLOOD_VALUE = 128
pixel_map = {}
regions = set()
ID = 0
pbar = tqdm(total=im.shape[0] * im.shape[1])
for y in range(im.shape[0]):
    for x in range(im.shape[1]):
        pbar.update(1)
        # Only run floodFill on pixels that haven't been filled
        if im[y, x] == 255:
            # create copy of image so floodFill doesn't change original
            im2 = im.copy()
            cv.floodFill(im2, None, (x, y), FLOOD_VALUE)
            region = PatchRegion(ID, pixels=np.argwhere(im2 == FLOOD_VALUE))
            for pixel in region.pixels:
                pixel_map[pixel] = region

            # visualize new region in image
            region.fill(im, FLOOD_VALUE + 1)

            # calculate edges for region
            region.set_edges(im, edge_map)
            regions.add(region)
            ID += 1
pbar.close()

print(f"Number of Regions: {len(regions)}")
for edge in tqdm(edge_map):
    neighbor = min(edge_map[edge], key=len)
    neighbor.pixels.append(edge)

color, NUM_COLORS = 0, 15
for region in tqdm(regions):
    region.fill(output, (color + 1) / (NUM_COLORS + 2) * 255)
    color = (color + 1) % NUM_COLORS

cv.imwrite("tree_patches.png", output)
cv.imshow("Patches", output)
cv.waitKey(0)
