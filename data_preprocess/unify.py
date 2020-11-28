from os import path
import cv2 as cv
from __future__ import division


class Unify:
    def __init__(self, width, height, image_out_path):
        self.width = width
        self.height = height
        self.image_out_path = image_out_path

    def image_resize(self, image_path):
        filename = path.basename(image_path)
        img = cv.imread(image_path)
        ori_height, ori_width = img.shape[0:2]
        if ori_width/ori_height > self.width/self.height:
            if ori_height > self.height:
                img_resize = cv.resize(img, (self.height / ori_height * ori_width, self.height),
                                       interpolation=cv.INTER_AREA)
            elif ori_height < self.height:
                img_resize = cv.resize(img, (self.height / ori_height * ori_width, self.height),
                                       interpolation=cv.INTER_CUBIC)
            else:
                img_resize = img
            start_width = round((ori_width - self.width) / 2)
            img_resize = img_resize[:, start_width:start_width + self.width]
        elif ori_width/ori_height < self.width/self.height:
            if ori_width > self.width:
                img_resize = cv.resize(img, (self.width, self.width / ori_width * ori_height),
                                       interpolation=cv.INTER_AREA)
            elif ori_width < self.width:
                img_resize = cv.resize(img, (self.width, self.width / ori_width * ori_height),
                                       interpolation=cv.INTER_CUBIC)
            else:
                img_resize = img
            start_height = round((ori_height - self.height) / 2)
            img_resize = img_resize[start_height:start_height + self.height, :]
        else:
            if ori_height > self.height:
                img_resize = cv.resize(img, (self.width, self.height),
                                       interpolation=cv.INTER_AREA)
            elif ori_height < self.height:
                img_resize = cv.resize(img, (self.width, self.height),
                                       interpolation=cv.INTER_CUBIC)
            else:
                img_resize = img
        # write
        cv.imwrite(path.join(self.image_out_path, filename), img_resize)
