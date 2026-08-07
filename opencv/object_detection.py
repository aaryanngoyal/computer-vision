import cv2 as cv
import time

# <----------Haar cascade classifier----------->
# <-----------loading cascade classifier----------->

# load pretrained cascade calssifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + ' haarcascade_eye.xml')

# check if cascade loaded successfully
if face_cascade.empty():
    print("error loading cascade classifier")
    exit()

# <----------basic object detection------------>

# load image
img = cv.imread("images/mori.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# load cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# detect faces
faces = face_cascade.detectMultiScale(
     gray, 
     scaleFactor=1.1,   # image size reduced at each scale
     minNeighbors=5,   # neighbours each candidate should have
     minSize=(30, 30),  # minimum object size
     flags=cv.CASCADE_SCALE_IMAGE
)

print(f"found {len(faces)} faces")

# draw bounding box around detected faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow('face detection', img)
cv.waitKey(0)

# <------------nested detection------------>

def detect(img, cascade):
    """
    Detect object using cascadec classifier
    """
    rects = cascade.detectMultiScale(img, 
                                     scaleFactor=1.3,
                                     minNeighbors=4,
                                     minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]    # Convert to (x1, y1, x2, y2)
    return rects

def draw_rects(img, rects, color):
    """
    Draw rectangle on image
    """
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


# load cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

# load and process image
img = cv.imread('images/face.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.equalizeHist(gray)

# detect faces
faces = detect(gray, face_cascade)
vis = img.copy()
draw_rects(vis, faces, (0, 255, 0))

# detect eyes within faces
if not eye_cascade.empty():
    for x1, y1, x2, y2 in faces:
        roi = gray[y1:y2, x1:x2]
        vis_roi = vis[y1:y2, x1:x2]
        eyes = detect(roi.copy(), eye_cascade)
        draw_rects(vis_roi, eyes, (255, 0, 0))

cv.imshow('face & eye detection', vis)
cv.waitKey(0)
cv.destroyAllWindows()

# <----------HOG detector---------->
# <---------people detection with HOG------------->

def inside(r, q):
    """
    Check if rectangle r is in rectangle q
    """
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return (
        rx > qx and 
        ry > qy and 
        rx + rw < qx + qw and 
        ry + rh < qy + qh
    )

def draw_detections(img, rects, thickness=1):
    """
    draw detection rectangles
    """
    for x, y, w, h in rects:
        # hog detectors returns slightly larger rectangles so shrinking 
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

img = cv.imread("images/p2.jpg")

# HOG descriptor
hog = cv.HOGDescriptor()

# set default people detector
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# detect people
found, weights = hog.detectMultiScale(img, 
                                      winStride=(4, 4),
                                      padding=(8, 8),
                                      scale=1.1)

# filter overlapping detections
found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and inside(r, q):
            break
    else:
        found_filtered.append(r)

print(f"found {len(found_filtered)} people from {len(found)} detections")

# draw all detections
draw_detections(img, found)

# highlight filtered detections
draw_detections(img, found_filtered, 3)

cv.imshow('people detected', img)
cv.waitKey(0)
cv.destroyAllWindows()

# <------------real time detection on video------------->

# initialize hog detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# open camera
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # resize for faster processing
    frame = cv.resize(frame, (640, 480))

    # measure detection time
    start_time = time.time()

    # detect people
    found, weights = hog.detectMultiScale(frame,
                                          winStride=(8, 8),
                                          padding=(8, 8),
                                          scale=1.05)

    elapsed_time = time.time() - start_time
    fps = 1.0 / elapsed_time

    # draw detections
    for x, y, w, h in found:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # display fps and count
    cv.putText(frame, f'People: {len(found)}', (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f'FPS: {fps:.1f}', (10, 70),
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('hog detection', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

# <-----------performance optimization----------->

img = cv.imread("images/face.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# resize for fast detection
scale = 0.5
small = cv.resize(gray, None, fx=scale, fy=scale)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 
                                   'haarcascade_frontalface_default.xml')

# detect on smaller images
faces = face_cascade.detectMultiScale(small, 1.1, 5)

# scale coordniates back to original
faces = [[int(x/scale), int(y/scale), int(w/scale), int(h/scale)] for (x, y, w, h) in faces]

# draw on original
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow('show', img)
cv.waitKey(0)