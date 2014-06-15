import sys
import numpy as np
import cv2

MAX_WIDTH, MAX_HEIGHT = 1024, 1024

def varied(roi):
    has_0, has_255 = False, False
    for i in roi:
        for j in i:
            if j == 0:
                has_0 = True
            else:
                has_255 = True
            if has_0 and has_255:
                return True
    return False

im = cv2.imread(sys.argv[1], cv2.COLOR_BGR2GRAY)
im_height, im_width, _ = im.shape

if im_height > im_width:
    new_height = MAX_HEIGHT
    new_width = int(im_width * (float(MAX_HEIGHT) / im_height))
else:
    new_width = MAX_WIDTH
    new_height = int(im_height * (float(MAX_WIDTH) / im_width))

print((new_width, new_height))

im = cv2.resize(im, (new_width, new_height))

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, bw_im = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(bw_im, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.cv.CV_RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

clean_contours = set()
for cnt in contours:
    [x, y, w, h] = cv2.boundingRect(cnt)
    if w * h > 3:
        clean_contours.add((x, y, w, h))

print(len(clean_contours))

for x, y, w, h in clean_contours:
    roi = bw_im[y:y+h, x:x+w]
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('norm', im)
cv2.waitKey()
