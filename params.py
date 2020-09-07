

# PATHS

face_detection_model = "./serialized/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
face_detection_protopath = "./serialized/face_detection_model/deploy.prototxt"
dataset = "./datasets"
training_set = "/datascience/datasets/faces/UTKFace_train"
test_set = "/datascience/datasets/faces/UTKFace_test"
face_embedding_model = "./serialized/face_embedding_model/openface.nn4.small2.v1.t7"
images = "./images"


# Resolution

face_resolution = (200, 200)
video_resolution = (1600, 1300)


# Face detector's confidence

confidence = 0.6


# Number of photos per person to take

nb_photos = 20
