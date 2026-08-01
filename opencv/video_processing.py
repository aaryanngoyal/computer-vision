import cv2 as cv
import numpy as np


# <--------- capturing from camera -------------->

cap = cv.VideoCapture(0) # captures video from system camera

# checks if camera opened successfully
if not cap.isOpened():
    print("cannot open camera")
    exit()

# read and display frame
while True:
    ret, frame = cap.read()

    if not ret:
        print("CANT RECEIVE FRAMES")
        break

    cv.imshow('camera', frame)

    if cv.waitKey(1) & 0xFF == ord('q'): # q to quit
        break

# <---------- reading video files ------------->

# open video file
cap = cv.VideoCapture('/home/aaryangoyal/Downloads/finallllllllllllll.mp4')

if not cap.isOpened():
    print('cannot open video file')
    exit()

# get video properties
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

print(f"fps : {fps}, size : {width}*{height}, frame count : {frame_count}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow('video', frame)

    if cv.waitKey(int(1000/fps)) & 0xFF == 27:
        break

# <------------ video writer -------------->

cap = cv.VideoCapture(0)

# get video properties
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = 20.0

# define codec and create videowriter
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, fps, (width, height))

if not out.isOpened():
    print("cannot open video writer")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # write frame to output video
    out.write(frame)
    cv.imshow('recorded', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

# <----------- edge detection -------------->

def nothing(x):
    pass

# create window with trackbar
cv.namedWindow('edge')
cv.createTrackbar('threshold1', 'edge', 1000, 1000, nothing)
cv.createTrackbar('threshold2', 'edge', 1000, 1000, nothing)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # threshold values from trackbar
    thres1 = cv.getTrackbarPos('threshold1', 'edge')
    thres2 = cv.getTrackbarPos('threshold2', 'edge')

    # canny edge detection
    edges = cv.Canny(gray, thres1, thres2, apertureSize=5)

    # create visualization
    vis = frame.copy() # copy of frame
    vis = np.uint8(vis / 2.0) # darken frmaes
    vis[edges != 0] = (0, 0, 255) # highlight edge with red

    cv.imshow('edge', vis)
 
    if cv.waitKey(5) & 0xFF == 27:
        break

# <------------ full video processing pipeline ------------->

def process_frame(frame) :

    """
    Apply multiple processing steps to frame
    """
    # resize for fast processing
    frame = cv.resize(frame, (640, 480))

    # gaussian blur
    blur = cv.GaussianBlur(frame, (5, 5), 0)

    # bgr to hsv
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # defining color range(here blue)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # create mask
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # mask to original frame
    result = cv.bitwise_and(frame, frame, mask=mask)

    return result, mask

# main loop
cap = cv.VideoCapture('/home/aaryangoyal/Downloads/finallllllllllllll.mp4')

# steup video writer
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('processed.mp4', fourcc, fps, (640, 480))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # process frame
    processed, mask = process_frame(frame)

    out.write(processed)

    # display
    cv.imshow('original', cv.resize(frame, (640, 480)))
    cv.imshow('processed', processed)
    cv.imshow('mask', mask)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"processed {frame_count} frames")

    if cv.waitKey(1) & 0xFF == 27:
        break

print(f"Total frames processed: {frame_count}")

# release resouces
cap.release()
out.release()
cv.destroyAllWindows()