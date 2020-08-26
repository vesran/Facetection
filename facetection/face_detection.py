import cv2
import numpy as np
import params


# Import model which detects faces
detector = cv2.dnn.readNetFromCaffe(params.face_detection_protopath,
                                    params.face_detection_model)


def find_faces(image):
    (h, w) = image.shape[:2]

    # construct a blob from the image
    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, params.face_resolution), 1.0, params.face_resolution,
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(image_blob)
    detections = detector.forward()

    face_coords = []
    for i in range(detections.shape[2]):
        # loop over the detections
        # extract the confidence (i.e., probability) associated with
        # the prediction
        # i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > params.confidence:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            width = end_x - start_x
            height = end_y - start_y
            max_length = max(width, height)
            face_coords.append([start_x, start_x + max_length, start_y, start_y + max_length, confidence])

    return face_coords
