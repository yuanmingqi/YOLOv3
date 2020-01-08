import tensorflow as tf
import numpy as np
import cv2

from model import YOLOv3

yolov3 = YOLOv3(
    anchors_file='./data/yolo_anchors.txt',
    num_classes=80,
    train_file='./data/train.npz',
    epochs=50,
    batch_size=1,
    lr=0.0001,
    lr_decay=0.000001,
    shuffle=True,
    repeat=1,
    snapshots=''
)

model_file = 'model.h5'
model = tf.keras.models.load_model(model_file)

img_file = './data/timg.jpg'
img = cv2.imread(img_file)
img = cv2.resize(img, (416, 416)) / 255.0
inputs = np.expand_dims(img, 0)

yolo_outputs = model.predict(inputs)

boxes, scores, classes = yolov3.inference(
    yolo_outputs,
    image_shape=[416, 416]
)

print(boxes)
print(scores)
print(classes)

for item in boxes:
    img = cv2.rectangle(img, (item[1], item[0]), (item[3], item[2]), (255, 0, 0), 5)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



