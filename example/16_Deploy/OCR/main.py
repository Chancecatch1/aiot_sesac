import easyocr
import cv2

reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

img = cv2.imread('test.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
result = reader.readtext(img)

for ocr in result:
    ps = ocr[0]
    s = (int(ps[0][0]), int(ps[0][1]))
    e = (int(ps[2][0]), int(ps[2][1]))
    img = cv2.rectangle(img, s, e, (0, 255, 0), 2)
    print(ocr[1])

cv2.imshow('asd', img)

cv2.waitKey(0)
cv2.destroyAllWindows()