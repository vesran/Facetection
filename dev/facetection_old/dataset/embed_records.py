import tensorflow as tf
import cv2

from dev.facetection_old.dataset.io_records import parse_data, read_utkface_tfrecord

training_file = './serialized/training_set.tfrecords'

init_ds = read_utkface_tfrecord(training_file, label='gender')
labelled_ds = init_ds.map(parse_data)
shuffled_ds = labelled_ds.shuffle(128)

embedder = cv2.dnn.readNetFromTorch("./serialized/face_embedding_model/openface.nn4.small2.v1.t7")


