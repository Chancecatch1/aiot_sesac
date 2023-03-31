import tensorflow as tf
import cv2
from rembg import remove
import numpy as np

loaded = tf.saved_model.load('./model/converea')
img = cv2.imread('./resample/0/img1296.jpg')
img = img[60:420, :420]
img = remove(img)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
img = cv2.resize(img, (150,150)) / 255.0# input데이터를 스케일링했기 때문에 예측할 때에도 스케일링 해야함


print(img.shape)

pred = loaded([img])
print(pred)
y = np.asarray(pred)
y = np.argmax(y, axis=1)

print(y)

