import cv2
import numpy as np
import yaml


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0

detector = cv2.dnn.readNetFromCaffe(params['face_detection_protopath'],
                                    params['face_detection_model'])


def detect_face_box(frame):
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, params['face_resolution']), 1.0, params['face_resolution'],
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    # extract the confidence (i.e., probability) associated with
    # the prediction
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]
    startX, startY, endX, endY = (0, 0, 0, 0)

    # filter out weak detections
    if confidence > params['confidence']:
        # compute the (x, y)-coordinates of the bounding box for
        # the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

    return confidence, startX, startY, endX, endY

