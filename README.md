# Malaria-Infected-Cell-Detection
Classifying malaria infected and uninfected cells, and visualizing ConvNet filter's performance.
## Work Machine
* Google Colaboratory
* 12GB NVIDIA Tesla K80 GPU
* 12Hr run time

[Learn More about Colaboratory](https://medium.com/@oribarel/getting-the-most-out-of-your-google-colab-2b0585f82403)

## Data
Dataset contains two classes of malaria cell images:
* Parasitized (Infected) Class: 13780 images
* Uninfected Class: 13780 images
* 27560 images altogether (337.08MB)

![Class 1: Parasitized (Infected)](https://user-images.githubusercontent.com/40007876/60862483-7e78dc00-a23f-11e9-8421-4445d5a2d6bc.png)
**Parasitized (Infected) Cell**
![Class 2: Uninfected](https://user-images.githubusercontent.com/40007876/60862499-83d62680-a23f-11e9-9ce9-a170b64f81a7.png)
**Uninfected Cell**

[Collect Data from Kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
 or [from US National Library of Medicine (original source)](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)
 
 ## Upload Data from Kaggle to Colab
 [Here is a step-by-step way in Stackoverflow](https://stackoverflow.com/questions/49310470/using-kaggle-datasets-in-google-colab)
 
 ## Required Libraries
 
```python
%matplotlib inline
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from zipfile import ZipFile
```

## Extract Zip file

```python
zip = ZipFile('cell-images-for-detecting-malaria.zip')
zip.extractall()
```

## Split File into Train and Validation sets

```python
pip install split-folders

import split_folders
split_folders.ratio('cell_images', output='output', seed=1337, ratio=(.7, .3, 0))
```

* 70% images for training (19290)
* 30% images for validation (8268)

## CNN Model

```python
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(128, 128, 3), activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(2, activation='softmax'))
```

#### Model Summary
* 2D Convolutional Layer 1: #Filter=32, Kernel Size=(3, 3)
* 2D Convolutional Layer 2: #Filter=32, Kernel Size=(3, 3)
* Pooling Layer 1: Pooling Type = Max Pooling, Pooling Window=(2, 2), Strides=2
* 2D Convolutional Layer 3: #Filters=64, Kernel Size=(3, 3)
* Pooling Layer 2: Pooling Type = Max Pooling, Pooling Window=(2, 2), Strides=2
* Dense Layer 1: #Unit=128
* Activation type: ReLU (for Conv, Pooling, Dense Layers)
* Softmax Activation in final layer


## Read Images from Directories

```python
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_data = train_datagen.flow_from_directory('output/train',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

validation_data = test_datagen.flow_from_directory('output/val',
                                            target_size = (128, 128),
                                            batch_size = 16,
                                            class_mode = 'categorical')
```

## Compile Model

```python
model.compile(Adam(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

## Model Checkpoint

```python
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)
```

## Train Model

```python
history = model.fit_generator(training_data,
                              steps_per_epoch= 1205,
                              epochs=30,
                              callbacks=[checkpointer],
                              validation_data=validation_data,
                              validation_steps=517)
```

* Train Steps per Epoch = (train data length)/(batch size) ~ 1205
* Validation Steps = (validation data length)/(batch size) ~ 517

## Accuracy vs Epoch Graph

```python
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
```

![Final_Accuracy](https://user-images.githubusercontent.com/40007876/60872144-04078680-a256-11e9-8256-14464e62381f.png)

* Train Accuracy = 96.75%
* Validation Accuracy = 95.98%

## Loss vs Epoch Graph

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
```

![Final_Loss](https://user-images.githubusercontent.com/40007876/60872161-0964d100-a256-11e9-8012-9c49513aa879.png)

* Train Loss = 9.36%
* Validation Loss = 12.14%

## Visualize Activations using Keract

[Learn Keract Here](https://github.com/philipperemy/keract)

**Output of Convolutional Layer 1:**

![conv2d_1](https://user-images.githubusercontent.com/40007876/60874015-23ec7980-a259-11e9-8bf3-4f6f2bf587f0.png)

**Output of Convolutional Layer 2:**

![conv2d_2](https://user-images.githubusercontent.com/40007876/60874025-28b12d80-a259-11e9-9a59-76f1563e712f.png)

**Output of Max Pooling Layer 1:**

![max_pooling2d_1](https://user-images.githubusercontent.com/40007876/60874002-1df69880-a259-11e9-9caa-c7f2fdf5b7ff.png)

**Output of Convolutional Layer 3:**

![conv2d_3](https://user-images.githubusercontent.com/40007876/60874032-2cdd4b00-a259-11e9-8ac4-42bf765be4a9.png)

**Output of Max Pooling Layer 2:**

![max_pooling2d_2](https://user-images.githubusercontent.com/40007876/60873889-eab40980-a258-11e9-95fd-9677a83fcb14.png)

[Learn More about Activation Visualization Here](https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0)

