import numpy as np
import cv2 as cv
from collections import defaultdict
from PatchRegion import PatchRegion
from tqdm import tqdm

# read in image of contours
image_contours = cv.imread("contours.png", 0)

# Find regions where there is a 2x2 grid of black
# pixels and set the top left corner to white. This cleans
# up the the intersection between edges formed from the
# contours.
print("Removing hidden pixels...")
filter_plus = np.ones((2, 2))
pbar = tqdm(total=(image_contours.shape[0] - 1) * (image_contours.shape[1] - 1))
for i in range(0, image_contours.shape[0] - 1):
    for j in range(0, image_contours.shape[1] - 1):
        window = image_contours[i : i + 2, j : j + 2]
        response = np.multiply(filter_plus, window)
        if np.all(response == 0):
            image_contours[i, j] = 255
        pbar.update(1)
pbar.close()

# Next, we identify all of the regions in our image.
# We use floodFill to find the segmented regions and we
# create a PatchRegion object that stores all of the
# interior pixels and edge pixels.
# edge_map maps pixels that are edges to the set of
# PatchRegions that have that edge.
print("Calculating PatchRegions...")
edge_map = defaultdict(set)
FLOOD_VALUE = 128
regions = set()
ID = 0
pbar = tqdm(total=image_contours.shape[0] * image_contours.shape[1])
for y in range(image_contours.shape[0]):
    for x in range(image_contours.shape[1]):
        pbar.update(1)
        # Only run floodFill on pixels that haven't been filled
        if image_contours[y, x] == 255:
            # create copy of image so floodFill doesn't change original
            image_contours2 = image_contours.copy()
            cv.floodFill(image_contours2, None, (x, y), FLOOD_VALUE)
            region = PatchRegion(ID, pixels=np.argwhere(image_contours2 == FLOOD_VALUE))

            # visualize new region in image
            region.fill(image_contours, 100)

            # calculate edges for region
            region.set_edges(image_contours, edge_map)
            regions.add(region)
            ID += 1
pbar.close()

print(f"Initial Number of Regions: {len(regions)}")

# allow each region to learn who its neighbors are
print("Updating neighbors...")
for region in regions:
    region.update_neighbors(edge_map)

# While there are PatchRegions that aren't "good", merge
# bad patches into good ones and remove them from our running
# set. Keep checking until no more patches are bad.
print("Merging regions...")
done = False
while not done:
    print("starting merge pass:", len(regions))
    done = True
    regions_to_remove = []
    for region in regions:
        if not region.is_good(image_contours):
            region.merge(edge_map)
            regions_to_remove.append(region)
            done = False
    for region in regions_to_remove:
        regions.remove(region)

print("Finished merging")
print(f"Final Number of Regions: {len(regions)}")

# visualize the outlines of the remaining patches
output = np.ones(image_contours.shape) * 255
for region in regions:
    region.trace(output, 0)

cv.imwrite(f"final_patches.png", output)
