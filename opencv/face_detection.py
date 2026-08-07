import cv2 as cv
import time

# <----------complete face and eye detection----------->

def detect(img, cascade):
    """
    detect faces and eyes using cascade classifier
    """
    rects = cascade.detectMultiScale(img, 
                                     minNeighbors=4,
                                     minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)

    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys
    import getopt

    # specify command line args
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])

    try:
        video_src = video_src[0]
    except:
        video_src = 0

    args = dict(args)
    cascade_fn = args.get('--cascade', cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    nested_fn = args.get('--nested-cascade', cv.data.haarcascades + "haarcascade_eye.xml")

    # load cascade
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    # capture video
    cap = cv.VideoCapture(video_src)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)

        # convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        # detection time
        t = time.time()

        # detect faces
        rects = detect(gray, cascade)
        vis = frame.copy()
        draw_rects(vis, rects, (0, 255, 0))

        # detect eyes
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = gray[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))

        dt = time.time() - t

        # display time elapsed
        cv.putText(vis, 'time: %.1f ms' % (dt*1000), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
        cv.imshow('detected', vis)

        if cv.waitKey(5) == 27:
            break

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

# <----------smile detection----------->

# load cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 
                                   'haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 
                                    'haarcascade_smile.xml')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # get roi for smile detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # detect smiles in lower half of face
        smiles = smile_cascade.detectMultiScale(
            roi_gray[h//2:, :],
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )

        # smile detection
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roi_color, (sx, sy + sh//2), (sx+sw, sy+sh + h//2), (0, 255, 0), 2)
            cv.putText(frame, "Smiling", (x, y-10),  cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv.imshow('smile detection', frame)

    if cv.waitKey(5) == 27:
        break

cap.release()
cv.destroyAllWindows()

# <-----------profile detection----------->

# load cascade
frontal_cascade = cv.CascadeClassifier(cv.data.haarcascades + 
                                      'haarcascade_frontalface_default.xml')
profile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 
                                      'haarcascade_profileface.xml')

img = cv.imread('images/profile.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect frontal face
frontal_face = frontal_cascade.detectMultiScale(gray, 1.1, 5)
print(f"no. of frontal faces: {len(frontal_face)}")

# detect profile(left)
profile_face = profile_cascade.detectMultiScale(gray, 1.1, 5)

# flip to detect profile(right)
gray_flip = cv.flip(gray, 1)
profile_flip = profile_cascade.detectMultiScale(gray_flip, 1.1, 5)

# flip coordinates back
width = img.shape[1]
profile_right = [[width-x-w, y, w, h] for (x, y, w, h) in profile_flip]

print(f"profile face(left): {len(profile_face)}")
print(f"profile faces(right): {len(profile_right)}")

# draw all detections
for (x, y, w, h) in frontal_face:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

for (x, y, w, h) in profile_face:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

for (x, y, w, h) in profile_right:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv.imshow('profile detection', img)
cv.waitKey(0)