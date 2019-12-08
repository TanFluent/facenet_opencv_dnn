import numpy as np
import cv2
import time

import pdb


face_image_path = './images/20191207105221-face_bbox.jpg'
model_path = './models/graph_final.pb'


# ------------------------------------------------------
def load_model(pb_model_path):
    net = cv2.dnn.readNetFromTensorflow(pb_model_path)
    return net


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# ------------------------------------------------------
""" Preprocess """
im = cv2.imread(face_image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
resized = cv2.resize(im, (160, 160), interpolation=cv2.INTER_LINEAR)
prewhitened = prewhiten(resized)
# HWC -> CHW
input_face_img = prewhitened.transpose([2, 0, 1])
# CHW -> NCHW
input_face_img = np.expand_dims(input_face_img, axis=0)

""" Load .pb model """
cvNet = load_model(model_path)

""" Forward """
cvNet.setInput(input_face_img)
stime = time.time()
cvOut = cvNet.forward()
etime = time.time()

dur = etime - stime
print('face feature extract dur: %f' % dur)

