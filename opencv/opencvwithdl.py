import cv2 as cv
import numpy as np
import time

# <---------YOLOv4 object detection (only detects person - broken)------------->

# load yolo network
net = cv.dnn.readNetFromDarknet('models/yolov4.cfg', 'models/yolov4.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# load class name
with open('models/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# video source
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    
    height, width = frame.shape[:2]

    # create blob from image
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                            swapRB=True, crop=False)

    # set input and run forward pass
    net.setInput(blob)

    # get outlayer names and forward pass
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    # process detections
    boxes = []
    class_ids = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:

                # scale bounding box back to original size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # get top-left corner
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # remove duplicates
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3)

    # draw detections
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_id]}: {confidences[i]:.2f}"
            color = (0, 255, 0)

            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv.imshow('object detection', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

# <-----------SSD object detection------------>

# load mobilenet ssd model
net = cv.dnn.readNetFromCaffe(
    'models/deploy.prototxt',
    'models/mobilenet_iter_73000.caffemodel'
)

# coco class names
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

img = cv.imread("images/dog.jpg")
height, width = img.shape[:2]

# prepare input
blob = cv.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)

# run detections
net.setInput(blob)
detections = net.forward()

# process detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        class_id = int(detections[0, 0, i, 1])

        # box coordinates
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (x1, y1, x2, y2) = box.astype("int")

        # draw detections
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(img, label, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv.imshow("img detection", img)
cv.waitKey(0)

# <----------image classification----------->

# load resnet model
net = cv.dnn.readNetFromCaffe(
    'models/ResNet-50-deploy.prototxt',
    'models/ResNet-50-model.caffemodel'    
)

# load imagenet classes
with open('models/classification_classes_ILSVRC2012.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

img = cv.imread("images/people.jpg")

# prepare inputs
blob = cv.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123), swapRB=False, crop=False)

# run inference
net.setInput(blob)
predictions = net.forward()

# get top 5 predictions
top5 = np.argsort(predictions[0])[::-1][:5]

for i, idx in enumerate(top5):
    label = classes[idx]
    confidence = predictions[0][idx]
    print(f"{i+1}, {label}: {confidence:.2f}")

# display
top_label = classes[top5[0]]
cv.putText(img, top_label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv.imshow('classification', img)
cv.waitKey(0)

# <---------video processing DNN----------->

# load model
net = cv.dnn.readNetFromCaffe(
    'models/deploy.prototxt',
    'models/mobilenet_iter_73000.caffemodel'
)

# capture video
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    height, width = frame.shape[:2]

    # prepare input
    blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # measure inference time
    start = time.time()
    net.setInput(blob)
    detections = net.forward()
    end = time.time()

    # process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # display fps
    fps = 1 / (end - start)
    cv.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('detection', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()