import tensorflow as tf
import cv2

from dev.facetection_old.dataset.io_records import parse_data, read_utkface_tfrecord


BATCH_SIZE = 32
EPOCHS = 5


#################################################################################
# Prepare data
#################################################################################

training_file = './serialized/training_set.tfrecords'

init_ds = read_utkface_tfrecord(training_file, label='gender')
labelled_ds = init_ds.map(parse_data)
shuffled_ds = labelled_ds.shuffle(128)

batched_ds = shuffled_ds.batch(BATCH_SIZE)
embedder = cv2.dnn.readNetFromTorch("./serialized/face_embedding_model/openface.nn4.small2.v1.t7")


def embed_images(images):
    face_blob = cv2.dnn.blobFromImages(images.numpy(), 1.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(face_blob)
    vec = embedder.forward()
    return vec


def tf_embed_images(images, label):
    return tf.py_function(embed_images, [images], Tout=tf.float32), label


embedded_ds = batched_ds.map(tf_embed_images)

#################################################################################
# Import FaceNet embedder
#################################################################################


class FaceNetEmbedder(tf.keras.layers.Layer):

    def __init__(self):
        super(FaceNetEmbedder, self).__init__()
        self.trainable = False
        self.embedder = cv2.dnn.readNetFromTorch("./serialized/face_embedding_model/openface.nn4.small2.v1.t7")

    def call(self, input, **kwargs):
        return tf.py_function(self.embed, [input], Tout=tf.float32)

    def embed(self, input):
        face_blob = cv2.dnn.blobFromImages(input.numpy(), 1.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(face_blob)
        vec = self.embedder.forward()
        return vec


# fn = FaceNetEmbedder()

# embedder = cv2.dnn.readNetFromTorch("./serialized/face_embedding_model/openface.nn4.small2.v1.t7")
# face_blob = cv2.dnn.blobFromImages(images[0].numpy(), 1.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)
# embedder.setInput(face_blob)
# vec = embedder.forward()

##########################################################

# from keras_facenet import FaceNet
# embedder = FaceNet()
#
# images = next(iter(batched_ds))
# # If you have pre-cropped images, you can skip the
# # detection step.
# embeddings = embedder.embeddings(images[0].numpy())

#################################################################################
# Training model
#################################################################################


class FNModel(tf.keras.layers.Layer):

    def __init__(self):
        super(FNModel, self).__init__()
        self.embedder = FaceNetEmbedder()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, input, **kwargs):
        x = self.embedder(input)
        x = self.d1(x)
        x = self.d2(x)
        return x


model = FNModel()
criteon = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

for epoch in range(EPOCHS):

    emp_mean = 0
    for step, (x, y) in enumerate(batched_ds):
        with tf.GradientTape() as tape:
            # [b, 1] where b is the size of a batch
            logits = model(x)
            # [b]
            logits = tf.squeeze(logits, axis=1)
            # [b] vs [b]
            loss = criteon(y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        emp_mean = (loss + step*emp_mean) / (step+1)
        if step % 10 == 0:
            print(f'{step} batches passed. Mean loss = {emp_mean}. Loss = {loss}')

    print(epoch, 'loss:', loss.numpy())



