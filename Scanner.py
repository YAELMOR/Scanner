# Rotem Drey 315585638
# Yael Hadad 315346718

import cv2
import numpy as np
import sys
import imutils
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local

# func that resized the image & used later
def resize_image(img_copy,plot):
    scale_precent = 100
    width = int(img_copy.shape[1] * scale_precent / 100)
    height = int(img_copy.shape[0] * scale_precent / 100)

    h = img_copy.shape[0]
    w = img_copy.shape[1]
    r = height / float(h)
    dim = (int(w * r), height)

    resized = cv2.resize(plot, dim, interpolation=cv2.INTER_AREA)
    return resized


def main(path_input, path_output):

    # Find the edges
    image = cv2.imread(path_input)
    ratio = image.shape[0] / 1050.0
    image_copy = image.copy()
    image = imutils.resize(image, height=1050)

    # Change to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blurring the image to find the edges easily
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_edged = cv2.Canny(image_gray, 50, 200)

    # find the contours in the edged image & keep the largest ones
    contours = cv2.findContours(image_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Approximation of the contour
    for n in contours:
        scope = cv2.arcLength(n, True)
        approx = cv2.approxPolyDP(n, 0.02 * scope, True)

        if len(approx) == 4: # need 4 points to know we found the screen
            screen_cntr = approx
            break
    # show the contours
    cv2.drawContours(image, [screen_cntr], -1, (0, 255, 0), 2)

    # get a top to down view of the original image
    res = four_point_transform(image_copy, screen_cntr.reshape(4, 2) * ratio)

    # Change the output image to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # use threshold to get the black and white scan
    TreshH = threshold_local(res, 11, offset = 10, method = "gaussian")
    plot = (res > TreshH).astype("uint8") * 255
    # show the original and scanned images
    cv2.imwrite(path_output, resize_image(image_copy,plot))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

