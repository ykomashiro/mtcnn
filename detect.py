from utils import *
from model import *


class Detect(object):
    def __init__(self, pnet, rnet, onet):
        self.P = pnet
        self.R = rnet
        self.O = onet

    def detect_face(self, image, minsize=30):
        h, w, _ = image.shape
        images, scales = generate_samples_from_image(image, minsize)
        ########################################################################
        # Note: first stage for PNet.
        ########################################################################
        total_bboxes = []
        total_bboxes_ref = []
        for img, scale in zip(images, scales):
            pnet_prob, pnet_loc = self.P(np.transpose(img, (0, 2, 1, 3)))
            pnet_prob = tf.transpose(pnet_prob, (0, 2, 1, 3))
            pnet_loc = tf.transpose(pnet_loc, (0, 2, 1, 3))
            original_bboxes = generate_original_boxes(pnet_loc.numpy(), scale)
            filter_mask = tf.argmax(pnet_prob, axis=-1)
            bboxes_tf = tf.boolean_mask(original_bboxes, filter_mask)
            bboxes_ref_tf = tf.boolean_mask(pnet_loc, filter_mask)
            local_bboxes, local_bboxes_ref = bboxes_nms(
                bboxes_tf.numpy(), bboxes_ref_tf.numpy(), 0.6)
            total_bboxes.append(local_bboxes)
            total_bboxes_ref.append(local_bboxes_ref)
        total_bboxes = np.concatenate(total_bboxes)
        total_bboxes_ref = np.concatenate(total_bboxes_ref)
        bboxes_pnet = bboxes_select(total_bboxes, total_bboxes_ref, 0.7)
        if bboxes_pnet.shape[0] == 0:
            return None
        bboxes_pnet, pad_bboxes_pnet = bboxes_clip(bboxes_pnet, w, h)
        images = crop_image(image, bboxes_pnet, pad_bboxes_pnet)
        ########################################################################
        # Note: second stage for RNet.
        ########################################################################
        rnet_prob, rnet_loc = self.R(np.transpose(images, (0, 2, 1, 3)))
        scores = rnet_prob[:, 1]
        filter_mask_rnet = scores > 0.7
        bboxes_ref_rnet_tf = tf.boolean_mask(rnet_loc, filter_mask_rnet)
        bboxes_rnet_tf = tf.boolean_mask(bboxes_pnet, filter_mask_rnet)
        bboxes_rnet = bboxes_select(
            bboxes_rnet_tf.numpy(), bboxes_ref_rnet_tf.numpy(), 0.7)
        if bboxes_rnet.shape[0] == 0:
            return None
        bboxes_rnet, pad_bboxes_rnet = bboxes_clip(bboxes_rnet, w, h)
        images = crop_image(image, bboxes_rnet, pad_bboxes_rnet, [48, 48])
        ########################################################################
        # Note: third stage for RNet.
        ########################################################################
        onet_prob, onet_loc, onet_landmark = self.O(
            np.transpose(images, (0, 2, 1, 3)))
        scores = onet_prob[:, 1]
        filter_mask_onet = scores > 0.7
        bboxes_ref_onet_tf = tf.boolean_mask(onet_loc, filter_mask_onet)
        bboxes_onet_tf = tf.boolean_mask(bboxes_rnet, filter_mask_onet)
        bboxes_onet = bboxes_select(
            bboxes_onet_tf.numpy(), bboxes_ref_onet_tf.numpy(), 0.7)
        if bboxes_onet.shape[0] == 0:
            return None
        bboxes_onet, pad_bboxes_onet = bboxes_clip(bboxes_onet, w, h)
        return bboxes_onet
