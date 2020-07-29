import numpy as np
import cv2
import time
import glob

CLASSES = [ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

img_files = []
imgdir = './dog'
for file in glob.glob(imgdir + "/*.jpg"):
    img_files.append(file)
for file in glob.glob(imgdir + "/*.png"):
    img_files.append(file)

threshold = 0.8
prototxt = './model/MobileNetSSD_deploy.prototxt.txt'
model = './model/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

prototxt2 = './model/resnet-50.prototxt'
model2 = './model/resnet-50.caffemodel'
net2 = cv2.dnn.readNetFromCaffe(prototxt2, model2)

input_imgfolder = False
srcvideo = 'out.2640'
cap = cv2.VideoCapture(srcvideo)
if not cap.isOpened():
    print("ERROR: Cannot open VideoCapture")
    input_imgfolder = True

frame_count, obj_count = 0, 0

while True:
    if input_imgfolder:
        if frame_count >= len(img_files):
            break
        imgfile = img_files[frame_count]
        frame = cv2.imread(imgfile)
    else:
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
            cv2.putText(frame2, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)

            # object classification
            crop_img = frame[startY:endY, startX:endX]
            scale_img = cv2.resize(crop_img, (224, 224))
            blob = cv2.dnn.blobFromImage(scale_img, 1.0, (224, 224), (104.0, 117.0, 123.0), False)
            net2.setInput(blob)
            probs = net2.forward()
            topv = sorted(probs[0], reverse=True)[0:5]
            topi = sorted(range(len(probs[0])), key=lambda k: probs[0][k], reverse=True)[0:5]
            result_txt = str(topi[0]+1) + ': ' + "{0:.4f}".format(topv[0])
            cx, cy = int((endX + startX)/2), int((endY + startY)/2)
            cv2.putText(frame2, result_txt, (startX, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 2)

            obj_count += 1

    frame_count += 1
    cv2.imshow("Frame", frame2)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
