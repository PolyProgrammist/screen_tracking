import cv2

cap = cv2.VideoCapture('sources/tv_on.mp4')

for image in cap:
    cv2.imshow(image)

