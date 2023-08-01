# -*- coding: utf-8 -*-


import cv2
import numpy as np

import darknet.darknet as dn


class Yolo(object):
    def __init__(self):
        self.data_file = "./models/vehicle-detector/coco.data"
        self.config_file = "./models/vehicle-detector/yolov4-leaky.cfg"
        self.weights = "./models/vehicle-detector/yolov4-leaky.weights"

        self.network, self.class_names, self.class_colors = dn.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            batch_size=1
        )

        self.width = dn.network_width(self.network)
        self.height = dn.network_height(self.network)
        self.darknet_image = dn.make_image(self.width, self.height, 3)

    # resize(Ow,Oh) --> original size(Nh,Nw)
    def backresize(self, Oh, Ow, bbx, Nh, Nw):
        Xt = Nw * 1.0 / Ow
        Yt = Nh * 1.0 / Oh
        x, y, w, h = bbx[:]

        xmin = max(np.floor((x - (w * 0.55)) * Xt).astype(int), 0)
        ymin = max(np.floor((y - (h * 0.55)) * Yt).astype(int), 0)
        xmax = min(np.ceil((x + (w * 0.55)) * Xt).astype(int), Nw)
        ymax = min(np.ceil((y + (h * 0.55)) * Yt).astype(int), Nh)

        return (xmin, ymin, xmax, ymax)

    def recoversize(self, detections, resizeimg, Newimg):
        Rh, Rw = resizeimg.shape[:2]
        Nh, Nw = Newimg.shape[:2]
        Res = []

        for detection in detections:
            bbx = (detection[2][0], detection[2][1], detection[2][2], detection[2][3])
            Res.append((detection[0], detection[1], self.backresize(Rh, Rw, bbx, Nh, Nw)))

        return Res

    def detection(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        dn.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
        detections = dn.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.45)

        Res = self.recoversize(detections, image_resized, image)
        # image = darknet.draw_boxes(detections, image_resized, self.class_colors)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Res
