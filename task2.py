import cv2
import os
import time
import numpy as np
from pathlib import Path
from task1 import hog
from task1 import contour_order
from task1 import draw_label
from task1 import clean_labels


# This function is used to detect and localize the building and directional signage
def locate_building_signage(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # Change color space from BGR to HSV
    h, s, v = cv2.split(hsv)  # Split channels of the image
    h1 = h * 0
    s1 = s * 0
    v1 = v * 0

    v_indices = np.where(v <= 100)  # Indices of the V channel that are below 100
    v[v_indices] = v[v_indices] * 0 + 255  # Make them white. This will reduce the shadows in the image
    s[v_indices] = s[v_indices] * 0 + 255  # Make them white. This will reduce the shadows in the image

    # Make a copy of the indices that are below threshold of 100
    h1[v_indices] = h[v_indices]
    s1[v_indices] = s[v_indices]
    v1[v_indices] = v[v_indices]

    img2 = cv2.merge((h1, s1, v1))  # This image will have shadows reduced
    img2 = cv2.erode(img2, (5, 5), iterations=29)  # Remove the background noise. In opencv erode works on background
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)  # Change color from HSV to BGR. In opencv cannot convert from HSV to Gray
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Change color from BGR to Gray
    ret, labels = cv2.connectedComponents(img2_gray)  # Find connected components
    labels = labels.T  # Transpose of the image
    labels = clean_labels(labels, 200)  # Remove the smaller and unwanted shaped connected components
    labels = labels.T  # Change the image back to its original orientation
    lbl_img = draw_label(labels)  # Draw the final connected components on black background
    lbl_img_gray = cv2.cvtColor(lbl_img, cv2.COLOR_BGR2GRAY)  # Change color from BGR to Gray
    ret, thresh = cv2.threshold(lbl_img_gray, 20, 255, cv2.THRESH_BINARY)  # Change the image to Binary using a threshold of 20

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)  # Find the connected components with more information
    roi_list = list()
    for i in range(1, nlabels):  # background is at the 0 index. That's why, loop starts from 1
        top_left = (stats[i][0], stats[i][1])  # Top left corner x and y
        width, height = stats[i][2], stats[i][3]
        bot_right = (stats[i][0] + width, stats[i][1] + height)  # Bottom right corner x and y
        area = height*width
        if 7.0 >= height/width >= 3.0 and area >= 10000:  # Filter based on area and aspect ratio
            roi_list.append((top_left, bot_right))
    return roi_list


# This function cut the bigger plate of numbers and returns a list of plates that contain each set of numbers
def cut_plate(plate):
    width = plate.shape[:2][1]
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  # Change color from BGR to Gray
    thresh = cv2.inRange(plate_gray, 90, 255)  # Change the image to Binary using a threshold of 90
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours to detect the digits
    hierarchy = hierarchy[0]
    list_of_plate = list()
    list_of_digit = list()
    for cmp in zip(contours, hierarchy):
        area = cv2.contourArea(cmp[0])
        if cmp[1][3] < 0 and 500 > area > 20:  # outer contours
            x, y, w, h = cv2.boundingRect(cmp[0])
            aspect_ratio = float(w)/h
            if 0.75 >= aspect_ratio >= 0.3 or 1.5 >= aspect_ratio >= 0.9:  # Filter contours based on aspect ration of digits and directional arrow
                list_of_digit.append((x, y, w, h))

    # This loop removes all the digits that are at approximately (+- 50) same y
    while len(list_of_digit) > 0:
        list_of_digit_at_same_height = list()
        dgt_y = list_of_digit[0][1]
        for dgt in list_of_digit:
            if (dgt_y - 50) <= dgt[1] <= (dgt_y + 50):  # Filter digits that are approximately at same y
                list_of_digit_at_same_height.append(dgt)

        # Crop the area of the bigger plate that contain the digits of approximately same y
        if len(list_of_digit_at_same_height) > 0:
            arr = np.array(list_of_digit_at_same_height)
            plate_x = np.min(arr[:, 0])
            plate_y = np.min(arr[:, 1])
            plate_height = np.max(arr[:, 3])
            p = plate[plate_y: plate_y + plate_height, plate_x: plate_x + width]  # Part of the image that have digits of same y
            list_of_plate.append(p)
        for dgt in list_of_digit_at_same_height:  # Remove the same y digits from the original list of digits
            list_of_digit.remove(dgt)
    return list_of_plate


# This function recognizes the digits
def read_plate(plate):
    svm = cv2.ml.SVM_load('trained_svm.dat')  # Load the pre trained model

    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  # Change color from BGR to Gray
    thresh = cv2.inRange(plate_gray, 90, 255)  # Change the image to Binary using a threshold of 90
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours of the digits
    hierarchy = hierarchy[0]

    big_contours = list()
    for cnt in zip(contours, hierarchy):
        if cnt[1][3] < 0 and 500 > cv2.contourArea(cnt[0]) > 20:  # outer contours
            _, _, w, h = cv2.boundingRect(cnt[0])
            aspect_ratio = float(w)/h
            if 0.75 >= aspect_ratio >= 0.3 or 1.5 >= aspect_ratio >= 0.9:  # Filter contours based on aspect ration of digits and directional arrow
                big_contours.append(cnt[0])

    big_contours.sort(key=lambda c: contour_order(c, plate.shape[1]))  # Sort contours based on same y from left to right

    plate_str = ''

    # This loop cuts the each digit from the number plate, resizes it to the required dimensions and then feed it to SVM for translation
    for i in range(len(big_contours)):
        mask = np.zeros_like(plate_gray)  # Black background of same dimensions as the number plate
        cv2.drawContours(mask, big_contours, i, 255, -1)  # Draw the digit on black background
        out = np.zeros_like(plate_gray)  # Another black background of same dimensions as the number plate
        out[mask == 255] = plate_gray[mask == 255]  # Shift the mask to second black background

        (y, x) = np.where(mask == 255)  # Find the x and y coordinate of each white pixel
        (topy, topx) = (np.min(y), np.min(x))  # Find the minimum of both x and y. This is the top left corner of digit
        (bottomy, bottomx) = (np.max(y), np.max(x))  # Find the maximum of both x and y. This is the bottom right corner of digit
        out = out[topy:bottomy+3, topx:bottomx+3]  # Add padding of 3 pixels to the sides of the digit
        _, thresh = cv2.threshold(out, 90, 255, cv2.THRESH_BINARY)  # Change it to binary
        dgt_resized = cv2.resize(thresh, (28, 40))  # Resize the digit to required dimensions

        hog_im = hog(dgt_resized)  # Find the descriptor of less dimensions using histogram of oriented gradients
        dgt_resized = np.array(hog_im, dtype='float32').reshape(-1, 64)
        result = svm.predict(dgt_resized)[1].ravel()  # Translate the digit

        # Decoding of left and right directional arrow
        if result == 10:
            plate_str += ' to the left'
        elif result == 11:
            plate_str += ' to the right'
        else:
            plate_str += str(int(result[0]))
    return plate_str


if __name__ == '__main__':
    t1 = time.time()
    program_folder = Path.cwd()  # Current directory of the program
    output_folder = program_folder.joinpath('output/task2')

    # For each run empty the task2 folder first in output directory
    for f in os.listdir(str(output_folder)):
        os.remove(str(output_folder.joinpath(f)))

    # Validation or Test images path
    images_folder = program_folder.joinpath('../test/task2')
    files = os.listdir(str(images_folder))

    for f in files:
        print(f)
        img = cv2.imread(str(images_folder.joinpath(f)))  # Read the image
        img_copy = np.copy(img)
        list_of_roi = locate_building_signage(img)  # Find the coordinates of the region of interest that contains the building numbers
        if len(list_of_roi) != 0:
            for roi in list_of_roi:
                (x1, y1), (x2, y2) = roi[0], roi[1]
                num_plate = img_copy[y1: y2, x1: x2]  # Crop the part of the image that contains building numbers

                fn = 'DetectedArea' + f[-6:]  # Last 6 characters of the current image file name
                fn2 = 'BuildingList' + f[-6:-4] + '.txt'  # Part of the file name that has number
                cv2.imwrite(str(output_folder.joinpath(fn)), num_plate)  # Write the image to task2 folder

                list_of_plates = cut_plate(num_plate)  # Cut the bigger number plate into small number plates
                text_file = open(str(output_folder.joinpath(fn2)), 'a')
                for j in range(len(list_of_plates)):
                    res = read_plate(list_of_plates[j])  # Translate each number plate
                    text_file.write('Building ' + res + '\n')  # Write the translated text file to task2 folder
                text_file.close()
    t2 = time.time()
    print('Processing Time: ', round((t2 - t1) * 100), 'ms')  # Total time for processing in milli seconds
