import sys
import numpy as np
import cv2
import sys
import statistics as st
import logging
from sklearn.cluster import MeanShift, estimate_bandwidth

from pathlib import Path

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ImageTransformer:
    def __init__(self, images):
        print("New image transformer instance made.")
        self.input_img_path = str(
            Path(__file__).resolve().parents[1] / "images" / images[0]
        )
        self.output_img_name = images[1]
        self.image = None
        self.original_image = None

    def transform_image(self):
        if self.read_image():
            self.structure_edges()
            self.refine_edges()
            # self.perform_meanshift()
        else:
            logger.info("Meanshift process returned None.")

    def read_image(self):
        """Reads an image to numpy array.

        :return: Numpy image array
        :rtype: _type_
        """
        # Loading image
        img = cv2.imread(self.input_img_path)
        if img is not None:
            logger.info("Reading image - Success.")
            self.image = img
            return True
        else:
            return False

    def refine_edges(self):
        """Refines edges of the image and turns the image into black & white."""
        # Apply thresholding to image
        img = cv2.threshold(np.uint8(self.image), 40, 255, cv2.THRESH_BINARY_INV)[1]
        morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        # Finding contours
        contours = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        # contours.sort(key=len, reverse=True)
        # Array that contains contours that will be removed
        noisy_contours = []
        sample = []
        # Gathering sample of contour areas
        for contour in cnts:
            sample.append(cv2.contourArea(contour))
        # Filtering contours that are below mean of the sample
        for contour in cnts:
            if cv2.contourArea(contour) < 1.5 * st.mean(sample):
                # Appending small contours to an array
                noisy_contours.append(contour)
        # Filtering out noise
        denoised = cv2.drawContours(
            morphed, noisy_contours, -1, color=(0, 0, 0), thickness=1
        )
        # Perform thresholding again
        inverted = cv2.threshold(denoised, 5, 255, cv2.THRESH_BINARY_INV)[1]
        # Applying smoothening
        median = cv2.medianBlur(inverted, 5)
        self.image = median
        logger.info("Refining edges - Success.")
        cv2.imwrite("edges.png", self.image)

    def structure_edges(self):
        # Loading model for edge detector
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection(
            "data-models/model.yml"
        )
        # Converting image colors
        src = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Detecting edges
        edges = edge_detector.detectEdges(np.float32(src) / 255.0)
        # Saving edge detected image to self
        self.image = 255 * edges
        logger.info("Structuring edges - Success.")

    def perform_meanshift(self):
        """Performs meanshift operation on the target image."""

        originShape = self.image.shape
        # Converting image into array of dimension [nb of pixels in originImage, 3]
        # based on r g b intensities
        flattened_img = np.reshape(self.image, [-1, 3])
        # Estimate bandwidth for meanshift algorithm
        bandwidth = estimate_bandwidth(flattened_img, quantile=0.12, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        # Performing meanshift on image
        ms.fit(flattened_img)
        # (r,g,b) vectors corresponding to the different clusters after meanshift
        labels = ms.labels_
        # Remaining colors after meanshift
        cluster_centers = ms.cluster_centers_
        self.image = cluster_centers[np.reshape(labels, originShape[:2])]
        logger.info("Meanshifting - Success.")


def main(args):
    it = ImageTransformer(args)
    it.transform_image()


if __name__ == "__main__":
    args = sys.argv
    # Check that both input and output image names are supplied.
    if len(args) == 3:
        main(args[1:])
    else:
        print(
            "Supply input and output image names as arguments. Ex. python main.py input.png output.png"
        )
