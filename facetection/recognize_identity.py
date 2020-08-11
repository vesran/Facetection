import cv2
import yaml
import imutils
import numpy as np

from facetection.utils.recognize_age import recognize_face_name
from facetection.utils.face_detection import detect_face_box


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0


def recognize_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_resolution = params['video_resolution']

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=video_resolution[1])
        confidence, startX, startY, endX, endY = detect_face_box(frame)
        if confidence > params['confidence']:
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Prediction
            color = (0, 0, 255)
            face = frame[startY:endY, startX:endX]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Predict name
            pred_name, proba = recognize_face_name(face)
            text = f'{pred_name} -- {proba}'
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_from_video('./images/friends_scene.mp4')
