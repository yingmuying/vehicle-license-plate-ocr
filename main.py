# -*- coding: utf-8 -*-

import cv2
import numpy as np

from lp_detect_align_tf import PlateDete
from lp_ocr import OcrRe
from vehicle_detect import Yolo


def main_video():
    yolo = Yolo()
    plate_dete = PlateDete()
    ocr_re = OcrRe()

    cap = cv2.VideoCapture("./samples/000.mp4")
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 640)

    cnt = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Can not Open Cam or Video is Over! ")
            break

        img_copy = img.copy()
        Res = yolo.detection(img_copy)

        if len(Res) == 0:
            print("No detection!")
            continue

        cnt += 1
        for i, re in enumerate(Res):
            xmin, ymin, xmax, ymax = (Res[i][2][:])

            car = img.copy()[ymin + 2:ymax - 2, xmin + 2:xmax - 2]
            Imglp, shapes = plate_dete.predict(car)

            if Imglp is None:
                print("No Plate!")
                continue

            lp_img = (Imglp * 255.0).astype(np.uint8)
            lp_str = ocr_re.lpdetection(lp_img, thresh=0.55)

            if len(lp_str) == 0:
                print("No LP Number!")
                continue

            print('Ocr LP Num: %s' % lp_str)

            cv2.putText(car, lp_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 3)
            cv2.namedWindow('lp_ocr', 0)
            cv2.imshow('lp_ocr', car)
            cv2.waitKey(1)

        cv2.namedWindow('image', 0)
        cv2.imshow('image', img_copy)
        cv2.waitKey(1)


if __name__ == '__main__':
    main_video()
