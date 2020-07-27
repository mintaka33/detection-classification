import numpy as np
import cv2
import time

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

threshold = 0.8
prototxt = 'E:/Data/model/mobilenet-ssd/MobileNetSSD_deploy.prototxt.txt'
model = 'E:/Data/model/mobilenet-ssd/MobileNetSSD_deploy.caffemodel'
srcvideo = 'test/dog.264'

cap = cv2.VideoCapture(srcvideo)
if not cap.isOpened():
    print("ERROR: Cannot open VideoCapture")
    exit()

net = cv2.dnn.readNetFromCaffe(prototxt, model)

frame_count, obj_count = 0, 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break;
    (srch, srcw, _) = frame.shape
    scale = srcw/1000.0
    w2, h2 = (1000, int(srch/scale))
    frame2 = frame.copy() #cv2.resize(frame, (w2, h2), cv2.INTER_AREA)
    (h, w) = frame2.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    obj_count = 0
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # draw mask
            cv2.rectangle(frame2, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame2, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            # crop image and save to file
            crop_img = frame[startY:endY, startX:endX]
            crop_name = 'frame' + str(frame_count) + '_' + 'obj' + str(obj_count) + '.png'
            cv2.imwrite(crop_name, crop_img)
            obj_count += 1
    frame_count += 1
    cv2.imshow("Frame", frame2)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
