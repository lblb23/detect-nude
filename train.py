import yaml
from keras.callbacks import ModelCheckpoint
from keras.callbacks.tensorboard_v2 import TensorBoard  # tensorboard_v1
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

if config['class_mode'] == '2classes':
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    loss = 'binary_crossentropy'
    class_mode = 'binary'

elif config['class_mode'] == '3classes':
    model.add(Dense(3))
    model.add(Activation('softmax'))

    loss = 'categorical_crossentropy'
    class_mode = 'categorical'

model.compile(loss=loss,
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    f"data_{config['class_mode']}/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode=class_mode)

validation_generator = test_datagen.flow_from_directory(
    f"data_{config['class_mode']}/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode=class_mode)

checkpointer = ModelCheckpoint(
    filepath=f"models/weights_{config['class_mode']}_test.hdf5", verbose=1,
    save_best_only=True)

tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
                 write_graph=True, write_grads=False, write_images=False,
                 embeddings_freq=0, embeddings_layer_names=None,
                 embeddings_metadata=None, embeddings_data=None,
                 update_freq='epoch')

model.fit_generator(
    train_generator,
    samples_per_epoch=8961,
    nb_epoch=50,
    validation_data=validation_generator,
    nb_val_samples=2097,
    callbacks=[checkpointer, tb])
