import numpy as np
import os
import cv2
import h5py
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .. import utils
from .descriptor_base import DescriptorBase


class Orb(DescriptorBase):
    def __init__(self, patch_size=65):
        super(Orb, self).__init__("data")
        self.patch_size = (int(patch_size), int(patch_size))
        self.orb = cv2.ORB.create(1500, nlevels=32)

    def describe(self, image):
        
        kp, desc = self.orb.detectAndCompute(image, None)
        desc = np.array(desc, dtype=np.float32)
        if desc.size <= 1:
            desc = np.zeros((1, 32))
        return desc

    def extract_patches(self, img, keypoints):
               patches = []
        height, width, _ = img.shape
        for kp in keypoints:
            mask = np.zeros((height, width), np.uint8)

            pt = (int(kp.pt[0]), int(kp.pt[1]))

            cv2.circle(mask, pt, int(kp.size), (255, 255, 255), thickness=-1)
            _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            contours = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x, y, w, h = cv2.boundingRect(contours[0][0])

            feat_patch = img[y:y+h, x:x+w]

            patches.append(feat_patch)
        return patches

   
