import numpy as np
import cv2
import time


prototxt = 'E:/Data/model/resnet-50/resnet-50.prototxt'
model = 'E:/Data/model/resnet-50/resnet-50.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt, model)

src_img = cv2.imread('frame0_obj0.png')
blob = cv2.dnn.blobFromImage(cv2.resize(src_img, (224, 224)), 0.007843, (224, 224), 127.5)
net.setInput(blob)
results = net.forward()

results[0].sort()
results_reversed = results[0][::-1]
top10 = results_reversed[0:10]
print(top10)

print('done')