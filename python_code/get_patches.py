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
pixel_map = {}
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
            for pixel in region.pixels:
                pixel_map[pixel] = region
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
            region.merge(edge_map, pixel_map)
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

print("Starting bad patch removal...")
VISUALIZE_BOX = True
VISUALIZE_MASK = True
for region in tqdm(regions):
    # form bounding box for the patch
    x, y, w, h = cv.boundingRect(np.expand_dims(np.array(region.pixels), axis=1))

    if VISUALIZE_BOX:
        output_copy = output.copy()
        temp = np.ones(output.shape, dtype=np.uint8) * 255
        region.fill(temp, 50)
        region.trace(temp, 150)
        cv.rectangle(output_copy, (y, x), (y + h, x + w), 0, 2)
        cv.imshow("Patch", output_copy)
        cv.waitKey(0)

    # create binary mask that is only one for pixels
    # that are actually in the patch
    mask = np.zeros((w, h), dtype=np.uint8)
    region_pixels = set(region.pixels)
    for i in range(w):
        for j in range(h):
            if (x + i, y + j) in region_pixels:
                mask[i, j] = 255

    if VISUALIZE_MASK:
        cv.imshow("Bounding Box", mask)
        cv.waitKey(0)

    # perform erosion-dilation on the mask and find the difference
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    diff = mask - opening
    # cv.imshow("Mask", mask)
    # cv.imshow("After", opening)
    # cv.imshow("Diff", diff)
    # cv.waitKey(0)
    loser_pixels_mask = np.argwhere(diff == 255)
    loser_pixels = set((pixel[0] + x, pixel[1] + y) for pixel in loser_pixels_mask)
    good_pixels = set(pixel for pixel in region.pixels if pixel not in loser_pixels)

    # Find all edge pixels that are bordered by ONLY loser pixels
    loser_edge_pixels = set()
    for pixel in region.edge_pixels:
        neighbors = [(pixel[0] + i, pixel[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
        if not any([neighbor in good_pixels for neighbor in neighbors]):
            loser_edge_pixels.add(pixel)
            del edge_map[pixel]

    # Find the new edge pixels. These are the neighbors of loser pixels
    # that aren't losers.
    new_edge_pixels = set()
    for loser_pixel in loser_pixels:
        neighbors = [(loser_pixel[0] + i, loser_pixel[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
        for neighbor in neighbors:
            if (
                neighbor[0] >= 0
                and neighbor[0] < image_contours.shape[0]
                and neighbor[1] >= 0
                and neighbor[1] < image_contours.shape[1]
                and neighbor not in loser_pixels
                and neighbor not in loser_edge_pixels
            ):
                new_edge_pixels.add(neighbor)

    region.edge_pixels = region.edge_pixels.union(new_edge_pixels)

    # Repeatedly iterate through the set of all loser pixels and
    # if they are next to real edge pixels, swap their roles
    adoption_agency = loser_pixels.union(loser_edge_pixels)
    done = False
    while not done:
        done = True
        pixels_to_remove = set()
        pixels_to_add = set()
        for pixel in adoption_agency:
            neighbors = [(loser_pixel[0] + i, loser_pixel[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
            for neighbor in neighbors:
                if (
                    neighbor[0] >= 0
                    and neighbor[0] < image_contours.shape[0]
                    and neighbor[1] >= 0
                    and neighbor[1] < image_contours.shape[1]
                    and neighbor in region.edge_pixels
                ):
                    done = False
                    pixels_to_add.add(neighbor)
                    pixels_to_remove.add(pixel)
                    region.edge_pixels.add(pixel)
                    break

        adoption_agency = adoption_agency.union(pixels_to_add).difference(pixels_to_remove)

    region.pixels += list(adoption_agency)

cv.imwrite(f"final_patches.png", output)
