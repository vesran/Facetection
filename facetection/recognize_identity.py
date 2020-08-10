import cv2
import pickle
import os, yaml
import imutils
import numpy as np


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0


# Load models and encoder
with open('./serialized/name_recognizer/svm.pkl', 'rb') as f:
    name_recognizer = pickle.load(f)
with open('./serialized/name_recognizer/le.pkl', 'rb') as f:
    le_names = pickle.load(f)
protoPath = params['face_detection_protopath']
modelPath = params['face_detection_model']
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch(params['face_embedding_model'])


def recognize_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_resolution = params['video_resolution']
    face_resolution = params['face_resolution']

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=video_resolution[1])
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, face_resolution), 1.0, face_resolution,
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
        # filter out weak detections
        if confidence > params['confidence']:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Use import !
            color = (0, 0, 255)
            face = frame[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            face_embedding = vec.flatten()

            # Predict
            probas = name_recognizer.predict_proba([face_embedding])[0]
            print(probas)
            j = np.argmax(probas)
            pred_name = le_names.classes_[j]
            text = f'{pred_name} -- {probas[j]}'
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


recognize_from_video('./images/friends_scene.mp4')
