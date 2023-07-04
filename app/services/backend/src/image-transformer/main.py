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
    def __init__(self, args):
        logger.info("Starting image transforming.")
        self.input_img_path = str(
            Path(__file__).resolve().parents[1] / "images" / args[0]
        )
        self.output_img_name = args[1]
        self.threshold = int(args[2])
        self.image = None
        self.original_image = None

    def transform_image(self):
        if self.read_image():
            self.structure_edges()
            self.refine_edges()
            # self.perform_meanshift()
        else:
            logger.error("Image could not be read.")

    def read_image(self):
        """Reads an image to numpy array.

        :return: Numpy image array
        :rtype: _type_
        """
        # Loading image (in grayscale)
        img = cv2.imread(self.input_img_path, 0)
        if img is not None:
            logger.info("Reading image - Success.")
            self.image = img
            return True
        else:
            return False

    def structure_edges(self):
        logger.info("Structuring edges...")
        # Loading model for edge detector
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection(
            "data-models/model.yml"
        )
        # Converting image colors
        src = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Detecting edges (pixel values have to be converted to 0 - 1)
        edges = edge_detector.detectEdges(np.float32(src) / 255.0)
        # Saving edge detected image to self (rescaling the pixel values)
        self.image = 255 * edges
        logger.info("Structuring edges - Success.")

    def refine_edges(self):
        """Refines edges of the image and turns the image into black & white."""
        logger.info("Refining edges...")
        # Apply thresholding to image (38)
        img = cv2.threshold(np.uint8(self.image), self.threshold,
                            255, cv2.THRESH_BINARY_INV)[1]
        # img = cv2.adaptiveThreshold(
        #    np.uint8(self.image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        morphed = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        # Finding contours
        contours = cv2.findContours(
            morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        # Array that contains contours that will be removed
        noisy_contours = []
        sample = []
        # Gathering sample of contour areas
        for contour in cnts:
            sample.append(cv2.contourArea(contour))
        # Filtering contours that are far below the mean of the sample
        for contour in cnts:
            if cv2.contourArea(contour) < 0.5 * st.mean(sample):
                # Appending small contours to an array
                noisy_contours.append(contour)

        # Filtering out noise by drawing the mask on the image
        cv2.drawContours(
            morphed, noisy_contours, -1, color=(255, 255, 255), thickness=2
        )
        # Applying smoothening
        median = cv2.medianBlur(morphed, 5)
        self.image = median
        logger.info("Refining edges - Success.")
        cv2.imwrite(self.output_img_name, self.image)

    def perform_meanshift(self):
        """Performs meanshift operation on the target image."""

        originShape = self.image.shape
        # Converting image into array of dimension [nb of pixels in originImage, 3]
        # based on r g b intensities
        flattened_img = np.reshape(self.image, [-1, 3])
        # Estimate bandwidth for meanshift algorithm
        bandwidth = estimate_bandwidth(
            flattened_img, quantile=0.12, n_samples=500)
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
    logger.error(args)
    it = ImageTransformer(args)
    it.transform_image()


if __name__ == "__main__":
    args = sys.argv
    # Check that both input and output image names and edge threshold are supplied.
    if len(args) == 4:
        main(args[1:])
    else:
        print(
            "Supply input, output image names and edge threshold (1-50) as arguments. Ex. python main.py input.png output.png 15"
        )
