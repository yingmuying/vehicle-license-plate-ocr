# coding: utf-8


import cv2
import numpy as np
import os
import tensorflow as tf

from src.label import Shape, writeShapes
from src.lp_utils import reconstruct
from src.utils import im2single


def find_files(directory):
    files = []
    for path, dirs, filelist in os.walk(directory):
        for filename in filelist:
            files.append(os.path.join(path, filename))
    return files


class PlateDete:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.pb_file = './models/lp-detector/lp_model.pb'  # 模型
        # with tf.Graph().as_default() as graph:
        #     self.build_from_pb()
        #     gpu_options = tf.GPUOptions(allow_growth=True)
        #     sess_config = tf.ConfigProto(gpu_options=gpu_options)
        #     self.sess = tf.Session(graph=graph, config=sess_config)
        #     self.sess.run(tf.global_variables_initializer())

        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.build_from_pb()
                self.sess.run(tf.global_variables_initializer())

    def build_from_pb(self):
        with tf.gfile.FastGFile(self.pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        self.input = tf.get_default_graph().get_tensor_by_name('input:0')  # 输入节点
        self.output = tf.get_default_graph().get_tensor_by_name('concatenate_1/concat:0')  # 输出节点

    def preprocess(self, carimage, net_step=2 ** 4):
        image = im2single(carimage)
        ratio = float(max(image.shape[:2])) / min(image.shape[:2])
        side = round(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 4)), 608)
        min_dim_img = min(image.shape[:2])
        factor = float(bound_dim) / min_dim_img
        w, h = (np.array(image.shape[1::-1], dtype=float) * factor).astype(int).tolist()
        w += (w % net_step != 0) * (net_step - w % net_step)
        h += (h % net_step != 0) * (net_step - h % net_step)
        Imgresized = cv2.resize(image, (w, h))  # , cv2.INTER_NEAREST
        return image, Imgresized

    def predict(self, carimage):
        image, Imgresized = self.preprocess(carimage)
        Reimg = Imgresized.copy()
        Reimg = Reimg.reshape((1, Reimg.shape[0], Reimg.shape[1], Reimg.shape[2]))

        model_preds = self.sess.run(self.output, feed_dict={self.input: Reimg})
        Yr = np.squeeze(model_preds)
        Locatelp, LlpImgs = reconstruct(image, Imgresized, Yr, out_size=(256, 96), threshold=0.5)

        if len(LlpImgs):
            Imglp = LlpImgs[0]
            Sh = Shape(Locatelp[0].pts)
        else:
            Imglp, Sh = None, None

        return Imglp, Sh
