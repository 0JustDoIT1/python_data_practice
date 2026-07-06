# deep learning
# layer를 추가한 이유: xor 문제를 해결
# layer를 깊게하는 이유: 추상화된 특징으로 정확한 특징 추출
# transfer lnearning을 하는 이유: 학습 시간을 단축(데이터가 작은 문제 해결) : 가중치가 초기화된 모델 (실제 이미지를 이용한 학습)
# transfer learning의 방법
# feature extraction, fine tunning, RoRa
# trainable-weight를 이용해서 제어
# vanishing gradient 기울기 소실 문제
#   deeper  : VGG16, VGG32, ResNet, DenseNet, NasNet(자동으로)
#   wider망 설계: GoogleNet, Inception, Xception(Depthwise seperable convolution),
#                 MobileNet, MixNet
#   deeper, wider, resolution을 고려 : EfficientNet
#   B0(resolution 224x22x) ~ B7
#   EfficientDet (object detection 객체 탐지)

# 이미지 classification, object detection(사각박스), segmentation(object 자체)
# object tracking(object 이동 탐지)으로 발전

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리
x_train = x_train[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

y_train.shape

from tensorflow.keras import layers, models
# 26x26x32
def create_cnn_model(): # 배치 사이즈
  inputs = layers.Input(shape=(28,28,1)) # 1장의 이미지
  x = layers.Conv2D(32, (3,3), activation = "relu", name = "conv1")(inputs)
  # 13x13x32
  x = layers.MaxPooling2D((2,2))(x) # 이미지 사이즈 줄임
  # 11x11x64
  x = layers.Conv2D(64, (3,3), activation = "relu", name = "conv2")(x)
  # 5x5x64
  x = layers.MaxPooling2D((2,2))(x)
  # 5x5x64
  x = layers.Flatten()(x)
  # (5x5x64) x 64
  x = layers.Dense(64, activation = "relu")(x)
  # 64x10
  # 배치사이즈 x 10
  outputs = layers.Dense(10, activation = "softmax")(x)
  model = models.Model(inputs, outputs)
  return model

model = create_cnn_model()
# one-hot-encoding된 데이터에 적용
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 3, validation_split = 0.1)

# GRAD-CAM 적용
import numpy as np
import matplotlib.pyplot as plt
import cv2

# GRAD-CAM 원리
# 마지막 feature-map(GAP : global average pooling)
# 10개가 target -> feature-map도 10개 생성
# channel별로 평균 -> 하나의 값 -> 예측 (가중치: 어떤 채널이 가장 크게 판단에 영향을 미쳤는가)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
  grad_model = tf.keras.models.Model(
      [model.input], # 입력데이터
      [model.get_layer(last_conv_layer_name).output, # featuremap
       model.output] # 최종판정(softmax의 결과)
  )
  with tf.GradientTape() as tape: # 미분
    # convolution 마지막 레이어의 feature-map
    # softmax의 예측값
    conv_outputs, predictions = grad_model(img_array)
    if pred_index is None:
      pred_index = tf.argmax(predictions[0]) # 가장 큰 인덱스
    class_channel = predictions[:, pred_index] # 가장 값이 큰 인덱스
  # 미분값이 크면 그 피쳐가 중요한 영향을 미쳤다는 것을 의미
  grads = tape.gradient(class_channel, conv_outputs)[0] # 미분값을 구함
  # GAP -> global average pooling
  pooled_grads = tf.reduce_mean(grads, axis = (0,1))
  conv_outputs = conv_outputs[0]
  # featuremap에 중요도를 곱해서 -> 결과를 합하면
  heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis = -1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= tf.reduce_max(heatmap) # 0~1 사이의 확률값으로 변환
  return heatmap.numpy()

def show_gradcam(image, model, last_conv_layer_name):
  # featuremap + gradient
  # 28x28x1 -> 모델에서는 4차원 1x28x28x1
  img_array = np.expand_dims(image, axis=0) # 1x28x28x1
  heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
  # opencv - 이미지처리, 영상처리
  # 이미지 사이즈 확대
  heatmap_resized = cv2.resize(heatmap, (28,28))
  heatmap_color = cv2.applyColorMap(
      # 0~1 확률값 -> 0 ~ 255
      # unsigned 0 ~ 255 -> 대응하는 컬러
      # 파란색, 빨간색
      np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
  # cvtColor 컴퓨터는 RGB를 사용, cv2는 BGR을 사용
  # cv2.COLORBGR2RGB 순서를 거꾸로 변경
  heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
  # 40%만 적용 -> 투명하게
  # 1x28x28x1 -> squeeze -> 28x28x1
  # GRAD-CAM이미지(heatmap) + 원본이미지(image)
  superimposed_img = heatmap_color * 0.4 + np.squeeze(
      image)[:, :, np.newaxis] * 255.0
  plt.figure(figsize = (10,4))
  plt.subplot(1,3,1)
  plt.title("Original")
  plt.imshow(np.squeeze(image), cmap = "gray")
  plt.axis("off")
  plt.subplot(1,3,2)
  plt.title("Grad-CAM Heatmap")
  plt.imshow(heatmap_resized, cmap = "jet") # heatmap 이미지
  plt.axis("off")
  plt.subplot(1,3,3)
  plt.title("Super imposed")
  plt.imshow(np.uint8(superimposed_img)) # 원본 + 히트맵 이미지
  plt.axis("off")
  plt.show()


index = 3
test_img = x_test[index]
true_label = np.argmax(y_test[index])
pred = model.predict(np.expand_dims(test_img, axis=0))
pred_label = np.argmax(pred[0])
print(f"예측: {pred_label}, 실제: {true_label}")
show_gradcam(test_img, model, last_conv_layer_name = "conv2")
# 뭘보고 0으로 판단했는가?

# imagedatagenerator
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

import numpy as np
np.random.seed(3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# data feeding(데이터 공급) + data qugmentatino(데이터 증강)
# 메모리에서 feeding
train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size = 128,
)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size = 128,
)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=15,
    epochs=30,
    validation_data=test_generator,
    validation_steps=5
)

evaluation = model.evaluate(test_generator)
print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

predictions = model.predict(test_generator)
predictions

# !pip install -U tensorflow-datasets protobuf # 버전 맞추기

import tensorflow_datasets as tfds
tfds.list_builders()

import tensorflow as tf
import tensorflow_datasets as tfds

ds, info = tfds.load(
    "mnist",
    split="train",
    shuffle_files=True,
    with_info=True
)

assert isinstance(ds, tf.data.Dataset)
print(ds)

print(info)

tfds.as_dataframe(ds.take(4), info)

fig = tfds.show_examples(ds, info)

ds, info = tfds.load(
    "CatsVsDogs",
    split="train",
    with_info=True)
fig = tfds.show_examples(ds, info)

# file에서 feeding
from zipfile import ZipFile
import numpy as np
with ZipFile("/content/drive/MyDrive/dataset/dogs_vs_cats-20230601T053814Z-001.zip", "r") as zip:
  zip.extractall()
  print("done")

with ZipFile("/content/dogs_vs_cats/train.zip", "r") as zip:
  zip.extractall()
  print("done")

with ZipFile("/content/dogs_vs_cats/test.zip", "r") as zip:
  zip.extractall()
  print("done")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cv2
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import os
PATH = "/content/train"
filename = os.listdir(PATH)
IMG_SIZE = 100

plt.figure(figsize = (10,10))
for i in range(1, 7):
  img_array = cv2.imread(os.path.join(PATH, filename[i]))
  resize_image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  plt.subplot(3,3,i)
  image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
  plt.axis("off")
  plt.imshow(image)

resize_image

resize_image[0].shape

training_data = []
x_train = []
y_train = []
for files in tqdm(os.listdir(PATH)): # progress bar(예측시간)
  try: # 파일 이름으로 부터 라벨링
    if files.find("cat") == -1: # 문자열 검색 실패 == -1
      category = 0 # dog
    else:
      category = 1 # cat
    # 학습은 흑백 이미지로 출력은 컬러 이미지로
    img_array = cv2.imread(os.path.join(PATH,files),
                           cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
    training_data.append([new_array, category])
    x_train.append([new_array])
    y_train.append([category])
  except Exception as e:
    pass

# 데이터를 ndarray로 변경
x_train = np.array(x_train)
y_train = np.array(y_train)

# 이미지 전처리 opencv
x_train.shape

x_train = x_train.reshape(25000, 100, 100, 1)

testing_data = []
PATH = "/content/test"
for files in tqdm(os.listdir(PATH)):
  try:
    img_array = cv2.imread(os.path.join(PATH, files),
                           cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    testing_data.append([new_array])
  except Exception as e:
    pass

import os
print(os.listdir("/content"))

print(os.listdir("/content/dogs_vs_cats"))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.3, random_state = 51)
X_train.shape

X_train = X_train/255.0
X_test = X_test/255.0

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(InputLayer(shape=(100,100, 1)))
model.add(Conv2D(32, kernel_size = (5,5), activation='relu'))
model.add(Conv2D(64, kernel_size = (5,5), activation='relu'))
# 1/4로 축소
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(0.001),
              metrics =['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
MODEL_DIR = '/content/drive/MyDrive/model/dogscats/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
modelpath="{epoch:02d}-{val_loss:.4f}.keras"
modelpos = os.path.join(MODEL_DIR, modelpath)
checkpointer = ModelCheckpoint(filepath=modelpos, monitor='val_loss', verbose=1,
                               save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


batch_size = 32
epochs = 30
history = model.fit(X_train, y_train, batch_size=batch_size, epochs = epochs,
                    verbose=1, validation_data = (X_test, y_test),
                    callbacks=[early_stopping_callback,checkpointer] )
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 시각화
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values)+1)
line1 = plt.plot(epochs, val_loss_values, label="Validation/Test Loss")
line2 = plt.plot(epochs, loss_values, label="Training Loss")
plt.setp(line1, linewidth=0.2, marker = "+", markersize = 10.0)
plt.setp(line2, linewidth=0.2, marker = "4", markersize = 10.0)
plt.grid(True)
plt.show()
# 과적합은 아님, 일반화 되어 잘 학습중

# 전이학습
# Xception 모델: Inception을 영향으로 wider하게 구성
# Depthwise seperable convolution을 구현
# 채널별로 convolution을 진행
from tensorflow import keras
base_model = keras.applications.Xception(
    weights = "imagenet", # imagent으로 사전에 학습된 모델
    input_shape = (100,100,3), # 컬러이미지를 입력
    include_top = False # head : dense망
    # cnn = cnn(convolution 연산) + FFNN
)
# feature-extraction만 실행
base_model.trainable = False # 가중치를 수정하지 않겠다 (fine-tunning하지 않음)
inputs = keras.Input(shape =(100,100,1))
# wider하게 모델이 구성
x = keras.layers.Concatenate()([inputs, inputs, inputs])
scale_layer = keras.layers.Rescaling(scale =1/127.5, offset = -1)
x = scale_layer(x) # 전처리
x = base_model(x, training=False) # CNN망을 통과
x = keras.layers.GlobalAveragePooling2D()(x) # 마지막에 GAP을 적용
x = keras.layers.Dropout(0.2)(x) # 과적합 방지
# 특징 추출 -> dense망으로 FFNN으로 데이터를 학습
outputs = keras.layers.Dense(2, activation = "softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# dogs vs cats -> 2진 분류
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy(from_logits = True),
    metrics = [keras.metrics.BinaryAccuracy()],
)
epochs= 10
history = model.fit(X_train, y_train, epochs = epochs, validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# wider, deeper, resolution 해상도 (이미지 사이즈)
# EfficientNet 3개의 중요요인을 balancing
# B0 ~ B7까지 모델 종류

from tensorflow import keras
base_model = keras.applications.EfficientNetB0(
    weights = "imagenet",
    input_shape = (100,100,3),
    include_top = False
)
base_model.trainable = False
inputs = keras.Input(shape=(100,100,1))
x = keras.layers.Concatenate()([inputs, inputs, inputs])
scale_layer = keras.layers.Rescaling(1./255)
x = scale_layer(x)
x = base_model(x, training = False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(2, activation = "softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy(from_logits = True),
    metrics = [keras.metrics.BinaryAccuracy()],
)
epochs= 10
history = model.fit(X_train, y_train, epochs = epochs, validation_data = (X_test, y_test))

# !pip uninstall -y scikit-learn
# !pip install scikit-learn==1.5.2
# # scikeras 별도의 패키지로 KerasClassifier가 이전
# !pip install scikeras[tensorflow]

from tensorflow import keras
from tensorflow.keras import layers

def create_model(dropout_rate=0.2, dense_units=128,
                 learning_rate = 1e-3, optimizer = "adam", activation = "relu"):
  base_model = kears.applications.EfficientNetV2S(
      weights = "imagenet",
      include_top = False,
      input_shape=(100,100,3)
  )
  base_model.trainable = False
  inputs = keras.Input(shape=(100,100,1))
  x = keras.layers.Concatenate()([inputs, inputs, inputs])
  x = layers.Rescaling(1./255)(inputs)
  x = base_model(x, training = False)
  x = layers.GlobalAberagePooling2D()(x)
  x = layers.Dropout(dropout_rate)(x)
  x = layers.Dense(dense_units, activation = activation)(x)
  outputs = layers.Dense(2, activation = "softmax")(x)
  model = keras.Model(inputs, outputs)

  if optimizer == "adam":
    opt = keras.optimizers.Adam(learning_rate = learning_rate)
  elif optimizer == "rmsprop":
    opt = keras.optimizers.RMSprop(
        learning_rate = learning_rate)
  else:
    opt = keras.optimizers.SGD(learning_rate = learning_rate)
  model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
  return model



training_data = []
x_train = []
y_train = []
for files in tqdm(os.listdir(PATH)): # progress bar(예측시간)
  try: # 파일 이름으로 부터 라벨링
    if files.find("cat") == -1: # 문자열 검색 실패 == -1
      category = 0 # dog
    else:
      category = 1 # cat
    # 학습은 흑백 이미지로 출력은 컬러 이미지로
    img_array = cv2.imread(os.path.join(PATH,files),
                           cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
    training_data.append([new_array, category])
    x_train.append([new_array])
    y_train.append([category])
  except Exception as e:
    pass

# 데이터를 ndarray로 변경
x_train = np.array(x_train)
y_train = np.array(y_train)

# # tuning -> 에러남
# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import LabelBinarizer
# import numpy as np
# # y_train = keras.utils.to_categorical(y_train, num_classes=2)
# model_wrapper = KerasClassifier(
#     model = create_model,
#     epochs = 10,
#     batch_size = 32,
#     verbose = 0
# )

# param_grid = {
#     "model__dropout_rate": [0.2, 0.3],
#     "model__dense_units": [64, 128],
#     "model__learning_rate": [1e-3, 1e-4],
#     "model__optimizer": ["adam", "rmsprop"],
#     "batch_size": [16,32]
# }

# grid = GridSearchCV(estimator = model_wrapper, param_grid = param_grid, cv = 3)
# grid_result = grid.fit(X_train, y_train)

# flower_photos

with ZipFile("파일경로", "r"):
  zip.extractall()
  print("done")

# 과제
# 데이터의 폴터 이름이 라벨이름
# VGG16, ResNet50, InceptionV3, DenseNet121로
# 모델을 구축하고 accuracy를 비교하시오.
# tf.keras.applications에 있는 모델을 사용하시오.
# colab 파일 제출