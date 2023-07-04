import os
import traceback

import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "../data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
CASCADE_PATH = "../data/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def get_landmarks(img):
    rects = cascade.detectMultiScale(img, 1.3, 5)
    x, y, w, h = rects[0]
    rect = dlib.rectangle(x, y, x + w, y + h)
    return np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])


def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 5, color=(0, 255, 255))
    return img


def get_lip_from_image(img, landmarks):
    x_min = 10000
    x_max = 0
    y_min = 10000
    y_max = 0

    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    print("x_min: ", x_min)
    print("x_max: ", x_max)
    print("y_min: ", y_min)
    print("x_max: ", y_max)

    roi_width = x_max - x_min
    roi_height = y_max - y_min

    if roi_width > roi_height:
        dst_len = 1.5 * roi_width
    else:
        dst_len = 1.5 * roi_height

    diff_x_len = dst_len - roi_width
    diff_y_len = dst_len - roi_height

    new_x = x_min
    new_y = y_min

    image_rows, image_cols, _ = img.shape
    if new_x >= diff_x_len / 2 and new_x + roi_width + diff_x_len / 2 < image_cols:
        new_x = new_x - diff_x_len / 2
    elif new_x < diff_x_len / 2:
        new_x = 0
    else:
        new_x = image_cols - dst_len

    if new_y >= diff_y_len / 2 and new_y + roi_height + diff_y_len / 2 < image_rows:
        new_y = new_y - diff_y_len / 2
    elif new_y < diff_y_len / 2:
        new_y = 0
    else:
        new_y = image_rows - dst_len

    roi = img[int(new_y): int(new_y + dst_len), int(new_x): int(new_x + dst_len), 0:3]
    return roi


def list_files(input_dir, output_dir, is_test=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    list_dirs = os.walk(input_dir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            print(os.path.join(root, d))
        for f in files:
            file_name = f.split('.')[0]
            file_type = f.split('.')[1]
            filepath = os.path.join(root, f)
            try:
                img = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)
                landmarks = get_landmarks(img)
                show_img = annotate_landmarks(img, landmarks)
                roi = get_lip_from_image(img, landmarks)

                roi_path = output_dir + "/" + file_name + "_mouth.png"

                if is_test:
                    cv2.imshow("keypoint", show_img)
                    cv2.waitKey(0)

                print("save image to file: ", roi_path)
                cv2.imwrite(roi_path, roi)
            except:
                print("processed error: ", filepath)
                traceback.print_exc()
                continue


list_files("../images/raw/cry", "../images/faces/cry")
list_files("../images/raw/face", "../images/faces/face")
list_files("../images/raw/pout", "../images/faces/pout")
list_files("../images/raw/smile", "../images/faces/smile")
