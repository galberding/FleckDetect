import caffe
import numpy as np
from random import randint

class DataAugmentation(caffe.Layer):

    def setup(self, bottom, top):
        print 'Create custom python layer'
        self.num = len(bottom)
        if self.num != 2:
            raise Exception("Need exactly two bottom blobs.")
        if len(top) != 2:
            raise Exception("Need exactly two top blobs.")


    def reshape(self, bottom, top):
        for i in range(self.num):
            top[i].reshape(*bottom[i].data.shape)

    def forward(self, bottom, top):
        rotation = randint(0, 1)
        flipping = randint(0, 1)
        gaussian = randint(0, 1)

        if rotation == 0 and flipping == 0 and gaussian == 0:
            for i in range(self.num):
                top[i].data[...] = bottom[i].data
        else:
            img = bottom[0].data
            img_gt = bottom[1].data
            if rotation != 0:
                img = np.rot90(img, k=2, axes=(2, 3))
                img_gt = np.rot90(img_gt, k=2, axes=(2, 3))
            if flipping != 0:
                img = np.flip(img, 0)
                img_gt = np.flip(img_gt, 0)
            if gaussian != 0:
                img = img + np.random.normal(0, 5, bottom[0].data.shape)
            top[0].data[...] = img
            top[1].data[...] = img_gt


    def backward(self, top, propagate_down, bottom):
        pass
