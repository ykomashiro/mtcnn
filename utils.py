# -*-coding: utf-8-*-
import sys
import numpy as np
import cv2
import tensorflow as tf


def resize_image(image, size=(12, 12)):
    return cv2.resize(image, size)


def preprocess_image(image):
    """Preprocess a image"""
    image_cp = np.copy(image).astype(np.float32)
    # regularize the image
    image_regular = (image_cp-127.5)/128
    # expand the batch_size dim
    image_expanded = np.expand_dims(image_regular, axis=0)
    return image_expanded


def cal_scales(minsize, factor=0.709, size=(12, 12)):
    minlen = np.min(size)
    s = 12.0/minsize
    minlen *= (s*factor)
    scales = [s]
    while (minlen >= 12):
        scales.append(s * np.power(factor, len(scales)))
        minlen *= factor
    return scales


def generate_samples_from_image(image, minsize=20, factor=0.709):
    height, weight, _ = image.shape
    scales = cal_scales(minsize, factor, size=(weight, height))
    images = []
    for scale in scales:
        w, h = int(np.ceil(weight * scale)), int(np.ceil(height * scale))
        images.append(preprocess_image(resize_image(image, size=(h, w))))
    return images, scales


def generate_original_boxes(bbox_reg, scale):
    # bboxes shape of (1,h,w,4)
    bboxes = np.zeros_like(bbox_reg, dtype=np.float32)
    _, h, w, _ = np.shape(bboxes)
    y, x = np.mgrid[0:h, 0:w]
    bboxes[0, :, :, 0] = x * 2 / scale
    bboxes[0, :, :, 1] = y * 2 / scale
    bboxes[0, :, :, 2] = (x * 2+12) / scale
    bboxes[0, :, :, 3] = (y * 2+12) / scale
    return tf.convert_to_tensor(bboxes)


def bboxes_iou(bboxes1, bboxes2, method="union"):
    """Computing iou between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])

    if method == "min":
        iou = int_vol / np.minimum(vol1, vol2)
    else:
        iou = int_vol / (vol1 + vol2 - int_vol)
    return iou


def bboxes_nms(bboxes, bboxes_ref, nms_threshold=0.5, method="union"):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones((bboxes.shape[0],), dtype=np.bool)
    for i in range(bboxes.shape[0]-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):], method)
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = overlap < nms_threshold
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):],
                                                 keep_overlap)
    idxes = np.where(keep_bboxes)
    return bboxes[idxes], bboxes_ref[idxes]


def bboxes_reg(bboxes, bboxes_ref):
    """boxes bounding regression.

    Args:
        bboxes ([array]): original boxes
        bboxes_ref ([array]): [description]

    Returns:
        [array]: [description]
    """
    bboxes = bboxes.astype(np.float32)
    w = bboxes[:, 2] - bboxes[:, 0] + 1
    h = bboxes[:, 3] - bboxes[:, 1] + 1
    bboxes[:, 0] += bboxes_ref[:, 0] * w
    bboxes[:, 1] += bboxes_ref[:, 1] * h
    bboxes[:, 2] += bboxes_ref[:, 2] * w
    bboxes[:, 3] += bboxes_ref[:, 3] * h
    return bboxes


def rec2square(bboxes):
    """Convert bbox to square."""
    h = bboxes[:, 3]-bboxes[:, 1]
    w = bboxes[:, 2]-bboxes[:, 0]
    l = np.maximum(w, h)
    bboxes[:, 0] = bboxes[:, 0]+w*0.5-l*0.5
    bboxes[:, 1] = bboxes[:, 1]+h*0.5-l*0.5
    bboxes[:, 2:4] = bboxes[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxes


def bboxes_clip(bboxes, w, h):
    """Clip bounding boxes

    Args:
        bboxes ([array]): shape of (N, 4)
        w ([int]): the width of input image
        h ([int]): the heigh of input image

    Returns:
        [tuple of array]: return the position of cliped boxes and
        padding for each axis
    """
    bboxes_cp = bboxes.copy()
    boxes_w = (bboxes[:, 2]-bboxes[:, 0]+1).astype(np.int32)
    boxes_h = (bboxes[:, 3] - bboxes[:, 0] + 1).astype(np.int32)
    bboxes_cp[:, 0] = np.maximum(bboxes[:, 0], 0)
    bboxes_cp[:, 1] = np.maximum(bboxes[:, 1], 0)
    bboxes_cp[:, 2] = np.minimum(bboxes[:, 2], w)
    bboxes_cp[:, 3] = np.minimum(bboxes[:, 3], h)

    pad_bboxes = np.zeros_like(bboxes, dtype=np.int32)
    pad_bboxes[:, 0] = bboxes_cp[:, 0] - bboxes[:, 0]
    pad_bboxes[:, 1] = bboxes_cp[:, 1] - bboxes[:, 1]
    pad_bboxes[:, 2] = bboxes[:, 2] - bboxes_cp[:, 2]
    pad_bboxes[:, 3] = bboxes[:, 3] - bboxes_cp[:, 3]
    pad_bboxes[pad_bboxes < 0] = 0
    return bboxes_cp.astype(np.int32), pad_bboxes


def crop_image(image, bboxes, pad_bboxes, size=[24, 24]):
    # example shape of [N,24,24,3]
    shape = [bboxes.shape[0]] + list(size) + [3]
    images = np.zeros(shape, dtype=np.float32)
    for idx in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[idx]
        padding = ((pad_bboxes[idx, 1], pad_bboxes[idx, 3]),
                   (pad_bboxes[idx, 0], pad_bboxes[idx, 2]),
                   (0, 0))
        temp_img = image[y1:y2, x1:x2, :]
        temp_img = np.pad(
            temp_img, padding, 'constant', constant_values=0)
        images[idx] = resize_image(temp_img, size=tuple(size))
    return (images-127.5)/128


def bboxes_select(bboxes, bboxes_ref, threshold=0.5, method="union"):
    """A series operation for boxes selected.

    Args:
        bboxes ([array]): Original locations for boxes, shape of (N, 4).
        bboxes_ref ([array): Regression locations for boxes, shape of (N, 4).
        threshold (float, optional): Defaults to 0.5.

    Returns:
        [array]: selected boxes.
    """
    bboxes, bboxes_ref = bboxes_nms(
        bboxes, bboxes_ref, threshold, method=method)
    if bboxes.shape[0] == 0:
        return bboxes
    bboxes = bboxes_reg(bboxes, bboxes_ref)
    bboxes = rec2square(bboxes.copy())
    return bboxes
