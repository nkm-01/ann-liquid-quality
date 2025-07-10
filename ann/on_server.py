import keras
from keras.preprocessing import image
import numpy as np

WIDTH = 140
HEIGHT = 200
def read_image(filename: str):
    im = image.load_img(filename, target_size=(HEIGHT, WIDTH))
    im_array = image.img_to_array(im) / 255
    return im_array

CORRECT = [[1]]
WRONG = [[0]]    

labels = np.array(CORRECT*40*4 +
                  WRONG*60*4 +
                  CORRECT*40*4 +
                  WRONG*40*4 +
                  CORRECT*60*4 +
                  WRONG*19*4 +
                  CORRECT*41*4 +
                  WRONG*20*4 +
                  CORRECT*70*4 +
                  WRONG*110*4
)

images = []
DATA_PATH = '../data'
import os
for filename in sorted(os.listdir(DATA_PATH)):
    if (not filename.endswith('.png')) or filename.startswith('.'):
        continue

    images.append(read_image(f'{DATA_PATH}/{filename}'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

def scheduler(epoch, lr):
    if epoch > 0:
        return lr * np.exp(-0.003 * epoch)
    
    return lr

learning_rate_callback = keras.callbacks.LearningRateScheduler(scheduler)

input = keras.Input((HEIGHT,WIDTH,3))

c1 = keras.layers.DepthwiseConv2D(3, depth_multiplier=6, activation='relu')(input)
c2 = keras.layers.DepthwiseConv2D((5,3), depth_multiplier=5, activation='gelu')(c1)
c2 = keras.layers.BatchNormalization()(c2)
p1 = keras.layers.MaxPool2D((2,2))(c2)

c3 = keras.layers.DepthwiseConv2D((5,3), depth_multiplier=4, activation='gelu')(p1)
c3 = keras.layers.BatchNormalization()(c3)
cp1 = keras.layers.Conv2D(128, (1,1), activation='linear')(c3)
p2 = keras.layers.MaxPool2D((4,4))(cp1)

f = keras.layers.Flatten()(p2)
d = keras.layers.Dropout(0.05)(f)
l = fc10 = keras.layers.Dense(3072, activation='relu')(d)
l = fc20 = keras.layers.Dense(1024, activation='relu')(l)
l = fc25 = keras.layers.Dense(512, activation='relu')(l)
# l = keras.layers.Dense(256, activation='relu')(l)
l = fc30 = keras.layers.Dense(128, activation='relu')(l)
l = out = keras.layers.Dense(1, activation='sigmoid')(l)
model = keras.Model(input, l)

loss = keras.losses.BinaryFocalCrossentropy()

model.compile(loss=loss, optimizer=keras.optimizers.Lion(learning_rate=1e-3), metrics=['mae', 'mse', 'accuracy'])

history1 = model.fit(
    np.array(X_train),
    y_train,
    epochs=10,
    batch_size=256,
    callbacks=[
        learning_rate_callback,
    ])

model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=1e-4, metrics=['mae', 'mse', 'accuracy']))

history2 = model.fit(
    np.array(X_train),
    y_train,
    epochs=60,
    batch_size=256,
    callbacks=[
        learning_rate_callback,
    ])

model.save('model.keras')
