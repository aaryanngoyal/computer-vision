import cv2 as cv

# loading and showing of image
img = cv.imread('images/haikyuu.jpg')

# processing of image
# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# edge detection
edges = cv.Canny(gray, 100, 200)

# saving the result
cv.imwrite('edges.jpg', edges)

#resize an image
resize1 = cv.resize(img, (1000, 1000)) # specific dimension

scale = 0.5 # using scale factor
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
resize2 = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)

fx, fy = 0.5, 0.5
resize3 = cv.resize(img, None, fx=fx, fy=fy, interpolation=cv.INTER_AREA)

# cropping images
# crop using slicing
cropped1 = img[100:200, 300:500]

# crop using roi
x, y, w, h = 100, 50, 300, 200
roi = img[y:y+h, x:x+w]

# rotation
rotate90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
rotate90C = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
rotate180 = cv.rotate(img, cv.ROTATE_180)

# rotate at angle
height, width = img.shape[:2]
# roataion matrix for 45 angle
center = (width // 2, height // 2)
angle = 45
scale = 1.0
rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)

rotate = cv.warpAffine(img, rotation_matrix, (width, height)) # applying rotation

# fliping image
fliph = cv.flip(img, 1)
flipv = cv.flip(img, 0)
flip_bold = cv.flip(img, -1)

# image filtering and smoothing
blur1 = cv.GaussianBlur(img, (5,5), 0) # less blur due to less kernel size
blur2 = cv.GaussianBlur(img, (15, 15), 0) # more blur due to large kernel size

median = cv.medianBlur(img, 5) # image blur
bilateral = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# cv.imshow('median', median)
# cv.imshow('bilateral', bilateral)
# cv.imshow('blur1', blur1)
# cv.imshow('blur2', blur2)
# cv.imshow('fliph', fliph)
# cv.imshow('flipv', flipv)
# cv.imshow('flip_bold', flip_bold)
# cv.imshow('rotate', rotate)
# cv.imshow('rotate1', rotation_matrix)
# cv.imshow('rotate90', rotate90)
# cv.imshow('rotate90c', rotate90C)
# cv.imshow('rotate180', rotate180)
# cv.imshow('cropped1', roi)
# cv.imshow('haikyuu', img)
# cv.imshow('resize1', resize1)
# cv.imshow('resize2', resize2)
# cv.imshow('resize3', resize3)
cv.waitKey(0)
cv.destroyAllWindows()