import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (256, 256))
    input_image = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)

    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    for kp in keypoints:
        y, x, c = kp
        if c > 0.3:
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 4, (0, 255, 0), -1)
    cv2.imshow('MoveNet Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
