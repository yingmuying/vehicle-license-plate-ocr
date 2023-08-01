import cv2
import numpy as np


def draw_label(I, l, color=(255, 0, 0), thickness=1):
    wh = np.array(I.shape[1::-1]).astype(float)
    tl = tuple((l.tl() * wh).astype(int).tolist())
    br = tuple((l.br() * wh).astype(int).tolist())
    cv2.rectangle(I, tl, br, color, thickness=thickness)


def draw_lp(Img, pts):
    assert (pts.shape[0] == 2 and pts.shape[1] == 4)

    ptxy = [xy for xy in zip(np.round(pts[0]).astype(int), np.round(pts[1]).astype(int))]

    mask = np.zeros(Img.shape, np.uint8)
    pts = np.array(ptxy, np.int32).reshape((-1, 1, 2))
    mask = cv2.polylines(mask, [pts], True, (0, 255, 255))
    mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # ret, conts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # opencv3
    conts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # opencv4
    x, y, w, h = cv2.boundingRect(conts[0])
    roi_lp = Img[y:y + h, x:x + w]

    # draw window
    # cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2) # hua ju xing kuang
    # cv2.polylines(Img, [pts], True, (0, 0, 255))               # si ge dian hua kuang
    return roi_lp


def draw_losangle(Img, pts, color=(1., 1., 1.), thickness=2):
    assert (pts.shape[0] == 2 and pts.shape[1] == 4)

    for i in range(4):
        pt1 = tuple(np.round(pts[:, i]).astype(int).tolist())
        pt2 = tuple(np.round(pts[:, (i + 1) % 4]).astype(int).tolist())
        cv2.line(Img, pt1, pt2, color, thickness)


def write2img(Img, label, strg, txt_color=(0, 0, 0), bg_color=(255, 255, 255), font_size=2):
    wh_img = np.array(Img.shape[1::-1])

    font = cv2.FONT_HERSHEY_SIMPLEX

    wh_text, v = cv2.getTextSize(strg, font, font_size, 3)
    bl_corner = label.tl() * wh_img

    tl_corner = np.array([bl_corner[0], bl_corner[1] - wh_text[1]]) / wh_img
    br_corner = np.array([bl_corner[0] + wh_text[0], bl_corner[1]]) / wh_img
    bl_corner /= wh_img

    if (tl_corner < 0.).any():
        delta = 0. - np.minimum(tl_corner, 0.)
    elif (br_corner > 1.).any():
        delta = 1. - np.maximum(br_corner, 1.)
    else:
        delta = 0.

    tl_corner += delta
    br_corner += delta
    bl_corner += delta

    tpl = lambda x: tuple((x * wh_img).astype(int).tolist())

    cv2.rectangle(Img, tpl(tl_corner), tpl(br_corner), bg_color, -1)
    cv2.putText(Img, strg, tpl(bl_corner), font, font_size, txt_color, 3)
