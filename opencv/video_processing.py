import cv2 as cv

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

cap.release()
cap.destroyAllWindows()