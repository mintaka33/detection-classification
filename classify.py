import numpy as np
import cv2
import time

labels = []
with open('resnet-50-label.txt', 'rt') as f:
    for line in f:
        strip_line = line.split(':')[1].strip().strip("',")
        labels.append(strip_line)


class ClassifyOut():
    def __init__(self, topv, topi):
        self.topv = topv
        self.topi = topi
        self.strline = self.toString()
    def toString(self):
        out = ''
        topstr = ''
        for i in range(len(self.topv)):
            strv = "{0:.4f}".format(self.topv[i])
            topstr += strv + ' : ' + labels[topi[i]] + '\n'
        out += topstr
        return out

prototxt = 'E:/Data/model/resnet-50/resnet-50.prototxt'
model = 'E:/Data/model/resnet-50/resnet-50.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt, model)

src_img = cv2.imread('frame0_obj0.png')
scale_img = cv2.resize(src_img, (224, 224))
blob = cv2.dnn.blobFromImage(scale_img, 1.0, (224, 224), (104.0, 117.0, 123.0), False)
net.setInput(blob)
probs = net.forward()


topv = sorted(probs[0], reverse=True)[0:5]
topi = sorted(range(len(probs[0])), key=lambda k: probs[0][k], reverse=True)[0:5]

result = ClassifyOut(topv, topi)

print(result.toString())

print('done')