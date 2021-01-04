import numpy as np


class PatchRegion:
    """
    Store the interior pixels, edge pixels, and neighboring patches for a patch.

    Instance variables:
        - self.pixels (list): List of pixels that are in the interior of the patch
        - self.edge_pixels (set): Set of pixels that define the edges of the patch
        - self.neighbors (set): Set of PatchRegions that are neighbors to this patch
        - self.ID (int): unique number identifying this patch
    """

    def __init__(self, ID, pixels=[]):
        self.pixels = [tuple(p) for p in pixels]
        self.edge_pixels = set()
        self.neighbors = set()
        self.id = ID

    def fill(self, image, color):
        """
        Set the pixels in the image corresponding to this patch's interior pixels
        to the given color.
        """
        image[tuple(np.array(self.pixels).T)] = color

    def trace(self, image, color):
        """
        Set the pixels in the image corresponding to this patch's edge pixels
        to the given color.
        """
        for pixel in self.edge_pixels:
            image[pixel[0], pixel[1]] = color
        return image

    def set_edges(self, image, edge_map):
        """
        Determine the edges of this patch based on an input image. Update a given
        map such that each edge pixel maps to a set of regions that includes this one.
        """
        for pixel in self.pixels:
            neighbors = [
                (pixel[0] + i, pixel[1] + j) for i in range(-1, 2) for j in range(-1, 2)
            ]
            for neighbor in neighbors:
                # A pixel is an edge if its within bound and is black
                if (
                    neighbor[0] >= 0
                    and neighbor[0] < image.shape[0]
                    and neighbor[1] >= 0
                    and neighbor[1] < image.shape[1]
                    and image[neighbor[0], neighbor[1]] == 0
                ):
                    self.edge_pixels.add(neighbor)

        for pixel in self.edge_pixels:
            edge_map[tuple(pixel)].add(self)

    def update_neighbors(self, edge_map):
        """
        Given a map of edge pixels to regions. Get all of the neighbors of this patch.
        """
        for pixel in self.edge_pixels:
            for neighbor in edge_map[tuple(pixel)]:
                if neighbor != self:
                    self.neighbors.add(neighbor)

    def merge(self, edge_map, pixel_map):
        """
        Merge this patch into its smallest neighbor
        """
        neighbor = min(self.neighbors, key=len)
        neighbor.pixels.extend(self.pixels)
        for pixel in self.pixels:
            pixel_map[pixel] = neighbor

        # Since we are removing a patch, edges will need to be updated
        for pixel in self.edge_pixels:
            edge_neighbors = edge_map[tuple(pixel)]
            # If removing this patch leave this edge pixel with only one region
            # convert the edge pixel to an interior pixel for the patch we are merging
            # into.
            if len(edge_neighbors) == 2 and neighbor in edge_neighbors:
                neighbor.pixels.append(pixel)
                # we must remove this pixel from the list of edge pixels of the neighbor
                # since it is now an interior pixel
                neighbor.edge_pixels.remove(pixel)
                # remove the pixel from the edge map since it is not an edge anymore
                del edge_map[tuple(pixel)]
            # Otherwise, the edges of this region become new edges of the neighbor
            # we are merging into.
            else:
                neighbor.edge_pixels.add(pixel)
                edge_neighbors.remove(self)
                edge_neighbors.add(neighbor)

        # reset instance variable of the region
        self.pixels = []
        self.edge_pixels = set()
        self.neighbors = set()

    def __len__(self):
        return len(self.pixels)

    def __repr__(self):
        return f"PatchRegion: {self.id}"

    def is_good(self, image):
        """
        Determine where or not this patch is good for a given image. The patch must
        have a certain number of pixels based on how far it is from the center. We
        asssume here that the center is more likely to have important details that
        justify having smaller patch regions.
        """
        center = np.average(self.pixels)
        image_center = np.round(np.array([image.shape[0], image.shape[1]]) / 2)
        dist = np.linalg.norm(image_center - center)
        SIZE_THRESHOLD_MIN, SIZE_THRESHOLD_MAX = 250, 5000
        MAX_DISTANCE = 400
        threshold = (
            SIZE_THRESHOLD_MIN
            + (SIZE_THRESHOLD_MAX - SIZE_THRESHOLD_MIN)
            * min(dist, MAX_DISTANCE)
            / MAX_DISTANCE
        )
        return len(self.pixels) > threshold
