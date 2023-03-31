from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 360,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip=True,
                                   vertical_flip = True,
                                   fill_mode = 'nearest')


train_generator = train_datagen.flow_from_directory(
                  'resample',
                  target_size = (150, 150),
                  batch_size = 16,
                  class_mode = 'sparse')

for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기:', data_batch.shape)
    print('배치 레이블 크기:', labels_batch.shape)
    break


inputs = keras.Input(shape=(150, 150, 3), name="img")
x = layers.Conv2D(16, 3, 2, padding = 'same', activation="relu")(inputs)
x = layers.Conv2D(32, 3, 1, padding ='same', activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, 1, activation="relu", padding="same")(x)
x = layers.Conv2D(64, 3, 2, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(4)(x)
outputs = layers.Softmax()(x)

model = keras.Model(inputs, outputs, name="converea")
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(train_generator, batch_size=16, epochs=2)
