import cv2
import numpy as np
import sys

"""
Usage
Change DATA_PATH
Key 'N' to binarize image
Key 'S' to warp and finalize image

Assumptions
1.) The document is light and the background is dark
2.) The document is fully contained in the image
3.) The document occupies a large portion of the image

Output
A rectified image of width 500px. Since we know the aspect ratio the height will be determined by that.
Aspect ratio is that of a letter document (8.5x11)
"""

DATA_PATH = "/home/david/PycharmProjects/opencv/course1/projects/document_scanner/"

# Use GrabCut to binarize the image. This works if the image is light and the background is dark.

# Convention of defining color in opencv is BGR
LIGHT_GREEN = [128, 255, 128]  # rectangle color
LIGHT_RED = [128, 128, 255]  # PR BG
BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

# Creating Dictionary
DRAW_BG = {'color': RED, 'val': 0}
DRAW_FG = {'color': GREEN, 'val': 1}
DRAW_PR_FG = {'color': LIGHT_GREEN, 'val': 3}
DRAW_PR_BG = {'color': LIGHT_RED, 'val': 2}

# Setting up flags
rect = (0, 0, 1, 1)
drawing = False  # flag for drawing curves
rectangle = False  # flag for drawing rect
rect_over = False  # flag to check if rect drawn
rect_or_mask = 100  # flag for selecting rect or mask mode
value = DRAW_FG  # drawing initialized to FG
thickness = 3  # brush thickness
rect_not_done = True


# Application Function on mouse
def onmouse(event, x, y, flags, param):
    global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over, rect_not_done

    # Draw Rectangle
    if (event == cv2.EVENT_LBUTTONDOWN) and rect_not_done:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle:
            img = img2.copy()
            cv2.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            rect_or_mask = 0

    elif (event == cv2.EVENT_LBUTTONUP) and rect_not_done:
        rectangle = False
        rect_not_done = False
        rect_over = True
        cv2.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # Draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        if not rect_over:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)


if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # Loading images if image is given by command line
    if len(sys.argv) == 2:
        filename = sys.argv[1]  # Using file for image
        print("Loading Image \n")
    else:
        print("No input image given,  so loading default image,  ../../data/images/hillary_clinton.jpg \n")
        print("Correct Usage: python grabcut.py <filename> \n")
        filename = DATA_PATH + 'test_img.jpg'

    img = cv2.imread(filename)
    img2 = img.copy()  # a copy of original image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
    output = np.zeros(img.shape, np.uint8)  # output image to be shown

    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onmouse)
    cv2.moveWindow('input', img.shape[1] + 10, 90)

    print(" Instructions: \n")
    print(" Draw a rectangle around the object using right mouse button \n")

    while 1:

        cv2.imshow('output', output)
        cv2.imshow('input', img)
        k = cv2.waitKey(1)

        # key bindings
        if k == 27:  # esc to exit
            break
        elif k == ord('0'):  # BG drawing
            print(" Using Red color,  >mark background regions with left mouse button \n")
            value = DRAW_BG
        elif k == ord('1'):  # FG drawing
            print(" Using Green color, >mark foreground regions with left mouse button \n")
            value = DRAW_FG
        elif k == ord('2'):  # PR_BG drawing
            print(" Using Light Red color, >mark probable Background regions with left mouse button \n")
            value = DRAW_PR_BG
        elif k == ord('3'):  # PR_FG drawing
            print(" Using Light Green color, >mark probable foreground regions with left mouse button \n")
            value = DRAW_PR_FG

        elif k == ord('s'):  # save image
            bar = np.zeros((img.shape[0], 5, 3), np.uint8)

            # Use findContour to find the contours of the binary image.
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(output, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

            # Use approxPolyDP to convert the contour to a rectangle. (finds the approximate polygon)
            epsilon = 0.1 * cv2.arcLength(biggest_contour, True)
            approx = cv2.approxPolyDP(biggest_contour, epsilon, True)
            print(f'The approximate polygon corners are: \n {approx}')

            # Create destination points for homography
            pts_dst = np.array([[500, 0], [0, 0], [0, 647], [500, 647]], dtype=float)

            # Calculate homography using 4-point correspondences and rectify.
            # Use 4 points from approxPolyDP and 4 points from 500px width and height (500px * aspect ratio(647px))
            h, status = cv2.findHomography(approx, pts_dst)

            # Warp source image to destination based on homography
            final_img = cv2.warpPerspective(img, h, (500, 647))

            # Final image window
            cv2.namedWindow('final')
            cv2.imshow('final', final_img)

            # Draw all the contours
            # nnn = cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

            cv2.imwrite(DATA_PATH + 'output_image.png', final_img)
            print(" Result saved as image \n")

        elif k == ord('r'):  # reset everything
            print("resetting \n")
            rect = (0, 0, 1, 1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            rect_not_done = True
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
            output = np.zeros(img.shape, np.uint8)  # output image to be shown
            print(__doc__)

        elif k == ord('n'):  # segment the image
            print(""" For finer touchups,  mark foreground and background after pressing keys 0-3
            and again press 'n' \n""")

            if rect_or_mask == 0:  # grabcut with rect
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1

            elif rect_or_mask == 1:  # grabcut with mask
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

        # Final mask is the union of definitely foreground and probably foreground
        # mask such that all 1-pixels (cv2.GC_FGD) and 3-pixels (cv2.GC_PR_FGD) are put to 1 (ie foreground) and
        # all rest are put to 0(ie background pixels)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

        # Copy the region to output
        output = cv2.bitwise_and(img2, img2, mask=mask2)

    cv2.destroyAllWindows()
