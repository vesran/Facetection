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


def tf_embed_images(images, labels):
    return tf.py_function(embed_images, [images], Tout=tf.float32), labels


embedded_ds = batched_ds.map(tf_embed_images)


#################################################################################
# Training model
#################################################################################

# model = tf.keras.Sequential()
# model.add(a := tf.keras.layers.Dense(64, activation='relu', input_shape=(BATCH_SIZE, 128)))
# model.add(b := tf.keras.layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#
# model.fit(embedded_ds, epochs=EPOCHS)


class CustomModel(tf.keras.layers.Layer):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.a = tf.keras.layers.Dense(64, activation='relu', input_shape=(BATCH_SIZE, 128))
        self.b = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, input, **kwargs):
        x = self.a(input)
        x = self.b(x)
        return x


model = CustomModel()
criteon = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for i in range(EPOCHS):
    for j, batch in enumerate(embedded_ds, start=1):
        y = batch[1]
        with tf.GradientTape() as tape:
            x = model(batch[0])
            x = tf.squeeze(x, axis=1)
            loss = criteon(x, tf.cast(y, dtype=tf.float32))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if j % 10 == 0:
            print(f'EPOCH {i} -- Batch {j} -- loss : {loss}')

