import tensorflow as tf

model = tf.keras.models.load_model('yolov3.h5')

inputs = tf.keras.Input([416, 416, 3])
outputs = model(inputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open('yolov3.tflite', 'wb').write(tflite_model)