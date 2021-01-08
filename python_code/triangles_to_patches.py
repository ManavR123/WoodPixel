import argparse
import json
import os
import shutil
from collections import defaultdict


import cv2 as cv
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from PatchRegion import PatchRegion


def main(inputfile, outputdir):
    # Reading in triangles
    im = cv.imread(inputfile, 0)

    # Create all the PatchRegions by using floodFill to identiy all of the
    # segmented regions in the image.
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
            if not region.is_good(im):
                region.merge(edge_map, pixel_map)
                regions_to_remove.append(region)
                done = False
        for region in regions_to_remove:
            regions.remove(region)
    print("Finished merging")
    print(f"Final Number of Regions: {len(regions)}")

    # For each pixel, add it to the smallest regiom it borders
    for edge in tqdm(edge_map):
        neighbor = min(edge_map[edge], key=len)
        neighbor.pixels.append(edge)

    # Create output json
    print("Creating output json...")
    if os.path.isdir(outputdir):
        shutil.rmtree(outputdir)
    os.mkdir(outputdir)
    patches_dir = os.path.join(outputdir, "patches")
    os.mkdir(patches_dir)
    patches = []
    for region in tqdm(regions):
        # form bounding box for the patch
        x, y, w, h = cv.boundingRect(np.expand_dims(np.array(region.pixels), axis=1))

        # create binary mask that is only one for pixels
        # that are actually in the patch
        mask = np.zeros((h, w), dtype=np.uint8)
        region_pixels = set(region.pixels)
        for i in range(w):
            for j in range(h):
                if (x + i, y + j) in region_pixels:
                    mask[j, i] = 255

        mask_path = os.path.join("patches", f"{region.id:08}.bmp")
        cv.imwrite(os.path.join(outputdir, mask_path), mask)

        entry = {
            "curves_top": "",
            "curves_bot": "",
            "curves_left": "",
            "curves_right": "",
            "curves_diag": "",
            "bounding_box": {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            "mask": mask_path,
            "sub_regions": "",
            "rectangular": "false",
            "target_index": "1",
            "coordinate": {
                "x": "0",
                "y": "0"
            }
        }
        patches.append(entry)
    # print(patches)
    with open(os.path.join(outputdir, "grid.json"), 'w') as f:
        f.write(json.dumps({"patches": patches}, indent=4))

    # Visualize all of the regions
    output = np.ones(im.shape, dtype=np.uint8) * 255
    color, NUM_COLORS = 0, 15
    for region in tqdm(regions):
        region.fill(output, (color + 1) / (NUM_COLORS + 2) * 255)
        color = (color + 1) % NUM_COLORS

    cv.imwrite(f"{inputfile}_patches.png", output)
    # cv.imshow("Patches", output)
    # cv.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to take an image segmented into regions and outputs a directory of the patch masks and a json to input into fit_patches."
    )
    parser.add_argument("--input", type=str, help="path to input file")
    parser.add_argument("--output", type=str, help="path to output directory")
    args = parser.parse_args()

    assert args.input and args.output

    main(args.input, args.output)
