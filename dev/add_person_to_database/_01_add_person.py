'''
The goal is to add the identity of a new person to the database.
The input should be the name of the person and a video or a stream of him/herself.
The main function should create a new folder containing images of the person's face in the following format :
    000_martin_31122020.jpg - 1st photo of Martin taken the 31st December 2020.

'''

from datetime import date
import os
import re
from glob import glob
import cv2
import yaml
import time
import imutils
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0


def load_models(verbose=False):
    # Load our serialized face detector from disk
    print("[INFO] loading face detector...") if verbose else 0
    protoPath = params['face_detection_protopath']
    modelPath = params['face_detection_model']
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...") if verbose else 0
    embedder = cv2.dnn.readNetFromTorch(params['face_embedding_model'])
    return detector, embedder


def extract_faces_from_stream(name, verbose=False, delay=0.5):
    # Create the folder of not exists
    path = os.path.join(params['datasets'], name)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # Clean directory if exists
        files = glob(os.path.join(path, '*'))
        for f in files:
            print(f"Removing {f}")
            os.remove(f)

    detector, embedder = load_models(verbose=verbose)

    print("[INFO] starting video stream...") if verbose else 0
    video_resolution = params['video_resolution']
    face_resolution = params['face_resolution']
    nb_photo_taken = 0

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # start the FPS throughput estimator
    fps = FPS().start()

    # loop over frames from the video file stream
    start_time = time.time()
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
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

            color = (0, 0, 255)
            if nb_photo_taken < params['nb_photos'] and time.time() - start_time > delay:
                color = (0, 255, 0)
                start_time = time.time()
                face = frame[startY:endY, startX:endX]
                filename = os.path.join(path,
                                        f"{str(nb_photo_taken).zfill(3)}_{name}_{re.sub('-', '', str(date.today()))}.jpg")
                cv2.imwrite(filename, face)
                print(f'[INFO] Writing photo {filename} to disk') if verbose else 0
                nb_photo_taken += 1

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # update the FPS counter
        fps.update()

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # stop the timer and display FPS information
    fps.stop()
    print(f"[INFO] {nb_photo_taken} photos have been saved in {path}") if verbose else 0
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed())) if verbose else 0
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) if verbose else 0

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    return path


def extract_faces_from_video(name, video_path, verbose=False, delay=0.5):
    # Create the folder of not exists
    path = os.path.join(params['datasets'], name)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # Clean directory if exists
        files = glob(os.path.join(path, '*'))
        for f in files:
            print(f"Removing {f}")
            os.remove(f)

    nb_photo_taken = 0
    cap = cv2.VideoCapture(video_path)
    detector, embedder = load_models(verbose=verbose)
    start_time = time.time()
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

            color = (0, 0, 255)
            if nb_photo_taken < params['nb_photos'] and time.time() - start_time > delay:
                color = (0, 255, 0)
                start_time = time.time()
                face = frame[startY:endY, startX:endX]
                filename = os.path.join(path,
                                        f"{str(nb_photo_taken).zfill(3)}_{name}_{re.sub('-', '', str(date.today()))}.jpg")
                cv2.imwrite(filename, face)
                print(f'[INFO] Writing photo {filename} to disk') if verbose else 0
                nb_photo_taken += 1

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return path


if __name__ == '__main__':
    data_path = extract_faces_from_video('Rachel', './images/rachel_alone.mp4', verbose=True, delay=1.0)
    assert len(glob(os.path.join(data_path, '*'))) == params['nb_photos']


