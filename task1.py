import cv2
import os
import time
import numpy as np
from pathlib import Path


# This function draws the connected components on black background
def draw_label(lbls):
    # Map component labels to hue val
    label_hue = np.uint8(179*lbls/np.max(lbls))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img


# This function removes the connected components that have less horizontal thickness than a specified threshold
def clean_labels(lbls, count_threshold):
    unique_labels = np.unique(lbls)  # To calculate number of connected components in the parameter lbls
    for i in range(1, len(unique_labels)):
        current_label_indices = np.where(lbls == unique_labels[i])  # Grab the indices of one component
        current_label_row_indices = current_label_indices[0]  # Number of indices in one row
        current_label_col_indices = current_label_indices[1]  # Number of indices in one column

        # Unique indices of one row. This is basically indices of indices.
        unique_indices, indices_index, indices_count = np.unique(current_label_row_indices, return_index=True, return_counts=True)
        less_count_indices = indices_index[np.where(indices_count <= count_threshold)]  # Indices that are below threshold
        less_count_row_indices = current_label_row_indices[less_count_indices]  # Row indices that are below threshold
        less_count_row_indices_count = indices_count[np.where(indices_count <= count_threshold)]  # Number of row indices that are below threshold

        # This loop changes all the indices of one component that are below threshold to 0
        for j in range(len(less_count_row_indices)):
            row_index = np.where(current_label_row_indices == less_count_row_indices[j])[0][0]
            row_value = current_label_row_indices[row_index]
            col_value = current_label_col_indices[row_index]
            lbls[row_value][col_value:col_value+less_count_row_indices_count[j]] = 0
    return lbls


# This function is used to detect and localize the building signage
def locate_building_signage(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # Change color space from BGR to HSV
    h, s, v = cv2.split(hsv)  # Split channels of the image
    h1 = h * 0
    s1 = s * 0
    v1 = v * 0

    v_indices = np.where(v <= 85)  # Indices of the V channel that are below 85
    v[v_indices] = v[v_indices] * 0 + 255  # Make them white. This will reduce the shadows in the image
    s[v_indices] = s[v_indices] * 0 + 255  # Make them white. This will reduce the shadows in the image

    # Make a copy of the indices that are below threshold of 85
    h1[v_indices] = h[v_indices]
    s1[v_indices] = s[v_indices]
    v1[v_indices] = v[v_indices]

    img2 = cv2.merge((h1, s1, v1))  # This image will have shadows reduced
    img2 = cv2.medianBlur(img2, 11)  # Remove the background noise

    img2 = cv2.erode(img2, (5, 5), iterations=3)  # Remove the background noise. In opencv erode works on background
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)  # Change color from HSV to BGR. In opencv cannot convert from HSV to Gray
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Change color from BGR to Gray

    ret, labels = cv2.connectedComponents(img2_gray)  # Find connected components
    labels = clean_labels(labels, 15)  # Remove the smaller and unwanted shaped connected components
    labels = labels.T  # Transpose of the image
    labels = clean_labels(labels, 15)  # Remove the smaller and unwanted shaped connected components
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
        if 0.7 >= height/width >= 0.5 and 22000 >= area >= 6000:  # Filter based on area and aspect ratio
            roi_list.append((top_left, bot_right))
    return roi_list


# This function changes the file name to numeric value which is used as the target value during the training of SVM
def get_label(fname):
    lbl_list = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Left', 'Right']
    for i in range(len(lbl_list)):
        if lbl_list[i] in fname:
            return i


# This function reads the digits and convert them into data set required for the training of SVM
def load_digits_data(p):
    digits_data = list()
    digits_target = list()

    list_of_folders = os.listdir(p)
    for folder in list_of_folders:
        folder = p.joinpath(folder)
        list_of_files = os.listdir(folder)
        for digit_file in list_of_files:
            digit_img = cv2.imread(str(folder.joinpath(digit_file)), cv2.IMREAD_GRAYSCALE)  # Read the image in Gray scale
            _, thresh = cv2.threshold(digit_img, 127, 255, cv2.THRESH_BINARY)  # Convert to binary
            hog_img = hog(thresh)  # Find the HOG
            img_label = get_label(digit_file)  # Convert the file name to numeric value
            digits_data.append(hog_img)
            digits_target.append(img_label)
    return np.array(digits_data, dtype='float32'), np.array(digits_target)  # Final data set


# This function finds the HOG
def hog(im):
    number_of_bins = 16
    gradient_x = cv2.Sobel(im, cv2.CV_32F, 1, 0)  # Horizontal gradient
    gradient_y = cv2.Sobel(im, cv2.CV_32F, 0, 1)  # Vertical gradient
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y)  # Combine two gradients
    bins = np.int32(number_of_bins * angle / (2 * np.pi))  # Make 16 bins from 0 to 360 degree angle

    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]  # Image is divided into 4 squares
    magnitude_cells = magnitude[:10, :10], magnitude[10:, :10], magnitude[:10, 10:], magnitude[10:, 10:]  # Image is divided into 4 squares

    list_of_histogram = list()

    # This loop calculates the histogram of each big square of the image
    for bin_cell, magnitude_cell in zip(bin_cells, magnitude_cells):
        bin_count = np.bincount(bin_cell.ravel(), magnitude_cell.ravel(), number_of_bins)
        list_of_histogram.append(bin_count)

    hist = np.hstack(list_of_histogram)  # Change the histogram into 16 X 4 = 64 dimension vector
    return hist


def train_svm():
    training_digits_folder = Path.cwd().joinpath('../Digits/augmented')
    data, targets = load_digits_data(training_digits_folder)
    svm = cv2.ml.SVM_create()  # Create the model
    svm.setKernel(cv2.ml.SVM_INTER)  # Choose the filter
    svm.train(data, cv2.ml.ROW_SAMPLE, targets)  # Train the SVM using each Row of the data set as one sample
    svm.save('trained_svm.dat')  # Save the trained model in current directory


# This function recognizes the digits
def read_plate(plate):
    svm = cv2.ml.SVM_load('trained_svm.dat')  # Load the pre trained model

    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  # Change color from BGR to Gray
    thresh = cv2.inRange(plate_gray, 150, 255)  # Change the image to Binary using a threshold of 150

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours of the digits
    hierarchy = hierarchy[0]

    big_contours = list()
    for cnt in zip(contours, hierarchy):
        if cnt[1][3] < 0 and cv2.contourArea(cnt[0]) > 100:  # outer contours
            big_contours.append(cnt[0])
    big_contours.sort(key=lambda c: contour_order(c, plate.shape[1]))  # Sort contours based on same y from left to right
    plate_str = ''

    # This loop cuts the each digit from the number plate, resizes it to the required dimensions and then feed it to SVM for translation
    for i in range(3):
        mask = np.zeros_like(plate_gray)  # Black background of same dimensions as the number plate
        cv2.drawContours(mask, big_contours, i, 255, -1)  # Draw the digit on black background
        out = np.zeros_like(plate_gray)  # Another black background of same dimensions as the number plate
        out[mask == 255] = plate_gray[mask == 255]  # Shift the mask to second black background
        (y, x) = np.where(mask == 255)  # Find the x and y coordinate of each white pixel
        (topy, topx) = (np.min(y), np.min(x))  # Find the minimum of both x and y. This is the top left corner of digit
        (bottomy, bottomx) = (np.max(y), np.max(x))  # Find the maximum of both x and y. This is the bottom right corner of digit
        out = out[topy-3:bottomy+3, topx-3:bottomx+3]  # Add padding of 3 pixels to the sides of the digit
        _, thresh = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)  # Change it to binary
        dgt_resized = cv2.resize(thresh, (28, 40))  # Resize the digit to required dimensions
        hog_im = hog(dgt_resized)  # Find the descriptor of less dimensions using histogram of oriented gradients
        dgt_resized = np.array(hog_im, dtype='float32').reshape(-1, 64)
        result = svm.predict(dgt_resized)[1].ravel()  # Translate the digit
        plate_str += str(int(result[0]))
    return plate_str


# Function arranges the contours from left to right that are approximately at same y value.
def contour_order(cnt, ncols):
    height_tolerance = 10
    x, y, _, _ = cv2.boundingRect(cnt)  # Find x and y coordinates of the bounding box of the contour
    same_height = (y // height_tolerance) * height_tolerance
    return same_height * ncols + x


if __name__ == '__main__':
    # Uncoment the below line if you want to train the new model
    # train_svm()
    t1 = time.time()
    program_folder = Path.cwd()  # Current directory of the program
    output_folder = program_folder.joinpath('output/task1')

    # For each run empty the task1 folder first in output directory
    for f in os.listdir(str(output_folder)):
        os.remove(str(output_folder.joinpath(f)))

    # Validation or Test images path
    images_folder = program_folder.joinpath('../test/task1')
    files = os.listdir(str(images_folder))

    list_of_plates = list()
    for f in files:
        print(f)
        img = cv2.imread(str(images_folder.joinpath(f)))  # Read the image
        img_copy = np.copy(img)
        list_of_roi = locate_building_signage(img)  # Find the coordinates of the region of interest that contains the building numbers
        if len(list_of_roi) != 0:
            for roi in list_of_roi:
                (x1, y1), (x2, y2) = roi[0], roi[1]
                num_plate = img_copy[y1: y2, x1: x2]  # Crop the part of the image that contains building number
                fn = 'DetectedArea' + f[-6:]  # Last 6 characters of the current image file name
                fn2 = 'Building' + f[-6:-4] + '.txt'  # Part of the file name that has number
                cv2.imwrite(str(output_folder.joinpath(fn)), num_plate)  # Write the image to task2 folder
                res = read_plate(num_plate)  # Translate each number plate
                text_file = open(str(output_folder.joinpath(fn2)), 'w')
                text_file.write('Building ' + res)  # Write the translated text file to task2 folder
                text_file.close()
    t2 = time.time()
    print('Processing Time: ', round((t2 - t1) * 100), 'ms')  # Total time for processing in milli seconds
