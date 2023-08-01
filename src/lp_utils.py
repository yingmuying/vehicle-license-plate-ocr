import cv2
import numpy as np
import time
from os.path import splitext

from src.label import Label
from src.projection_utils import getRectPts, find_T_matrix
from src.utils import getWH, nms


class DLabel(Label):

    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)


def reconstruct(Imgorig, Imgresized, Y, out_size, threshold=.8):
    net_stride = 2 ** 4
    side = ((208. + 40.) / 2.) / net_stride  # 7.75

    Probs = Y[..., 0]
    Affines = Y[..., 2:]

    # rx, ry = Y.shape[:2]
    # ywh = Y.shape[1::-1]
    # imgwh = np.array(Imgresized.shape[1::-1], dtype=float).reshape((2,1))

    xx, yy = np.where(Probs > threshold)

    WH = getWH(Imgresized.shape)
    MN = WH / net_stride

    vxx = vyy = 0.5  # alpha

    base = lambda vx, vy: np.matrix([[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
    labels = []

    for i in range(len(xx)):
        y, x = xx[i], yy[i]
        affine = Affines[y, x]
        prob = Probs[y, x]

        mn = np.array([float(x) + .5, float(y) + .5])

        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0.)
        A[1, 1] = max(A[1, 1], 0.)

        pts = np.array(A * base(vxx, vyy))  # *alpha
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2, 1))

        pts_prop = pts_MN / MN.reshape((2, 1))

        labels.append(DLabel(0, pts_prop, prob))

    final_labels = nms(labels, .2)
    ImgLps = []

    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for i, label in enumerate(final_labels):
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate((label.pts * getWH(Imgorig.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)
            Imglp = cv2.warpPerspective(Imgorig, H, out_size, borderValue=.0)

            ImgLps.append(Imglp)

    return final_labels, ImgLps


# detect_lp(self.wpod_net, im2single(Ivehicle), bound_dim, 2 ** 4, (240, 80), 0.5)
def detect_lp(model, image, max_dim, net_step, out_size, threshold):
    min_dim_img = min(image.shape[:2])
    factor = float(max_dim) / min_dim_img

    w, h = (np.array(image.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    w += (w % net_step != 0) * (net_step - w % net_step)
    h += (h % net_step != 0) * (net_step - h % net_step)

    Imgresized = cv2.resize(image, (w, h))  # , cv2.INTER_NEAREST
    Reimg = Imgresized.copy()
    Reimg = Reimg.reshape((1, Reimg.shape[0], Reimg.shape[1], Reimg.shape[2]))

    start = time.time()
    Yr = model.predict(Reimg)
    # print("YrYrYrYrYrYr", type(Yr), Yr.shape)
    Yr = np.squeeze(Yr)
    elapsed = time.time() - start
    Locatelp, ReimgLps = reconstruct(image, Imgresized, Yr, out_size, threshold)

    return Locatelp, ReimgLps, elapsed
