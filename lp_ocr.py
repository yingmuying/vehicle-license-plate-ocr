# -*- coding: utf-8 -*-


import cv2
import numpy as np

import darknet.darknet as dn
from src.label import dknet_label_conversion
from src.utils import nms


class OcrRe(object):
    def __init__(self):
        self.data_file = './models/ocr/ocr-net.data'
        self.config_file = './models/ocr/ocr-net.cfg'
        self.weights = './models/ocr/ocr-net.weights'

        self.network, self.class_names, self.class_colors = dn.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            batch_size=1
        )

        self.width = dn.network_width(self.network)
        self.height = dn.network_height(self.network)
        self.darknet_image = dn.make_image(self.width, self.height, 3)

    def gamma_trans(self, image, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(image, gamma_table)

    def lpdetection(self, image, thresh=0.35):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        dn.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
        detections = dn.detect_image(self.network, self.class_names, self.darknet_image, thresh=thresh)

        lp_str = ""
        if len(detections):
            L = dknet_label_conversion(detections, self.width, self.height)
            L = nms(L, 0.45)

            L.sort(key=lambda x: x.tl()[0])
            lp_str = ''.join([chr(l.cl()) for l in L])

        return lp_str
