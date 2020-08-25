from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dropout, Activation
from keras import Sequential
from keras import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import keras


from dev.facetection_old.dataset.io_records import parse_data, read_utkface_tfrecord

BATCH_SIZE = 32
EPOCHS = 5
num_classes = 110


#################################################################################
# Prepare data
#################################################################################

training_file = './serialized/training_set.tfrecords'

init_ds = read_utkface_tfrecord(training_file)

keep_age = lambda x: parse_data(x, label='age')
labelled_ds = init_ds.map(keep_age)
shuffled_ds = labelled_ds.shuffle(128)


def pad_y(features, y):
    return features, tf.one_hot(y, num_classes)


shuffled_ds = shuffled_ds.map(pad_y)

batched_ds = shuffled_ds.batch(BATCH_SIZE)


def resize_images(features, y):
    return tf.image.resize(features, (224, 224)), y


batched_ds = batched_ds.map(resize_images)


################################################################################
# Model
################################################################################


# VGG-Face model
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# pre-trained weights of vgg-face model.
# you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
# related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
model.load_weights('./serialized/face_embedding_model/vgg_face_weights.h5')

for layer in model.layers[:-7]:
    layer.trainable = False

base_model_output = Sequential()
base_model_output = Convolution2D(num_classes, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

age_model = Model(inputs=model.input, outputs=base_model_output)

checkpointer = ModelCheckpoint(filepath='age_model.hdf5',
                               monitor="val_loss", verbose=1, save_best_only=True, mode='auto')
age_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

scores = []

for i in range(EPOCHS):
    print("epoch ", i)
    score = age_model.fit(batched_ds, epochs=1, callbacks=[checkpointer])
    scores.append(score)
    print(score)

