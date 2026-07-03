# !pip uninstall -y scikit-learn
# !pip install scikit-learn==1.5.2

# !pip install scikeras[tensorflow]

# 한글을 지원하지 않음
# 환경제어 : matplotlibrc 파일, plt.rcParams, plt.rc()
# 현재 디렉토리, 설치된 디렉토리
import matplotlib.pyplot as plt # 시각화 script
# !apt-get update -qq # ubuntu 설치 명령어
# !apt-get install fonts-nanum* -qq # 나눔폰트 / 시스템관리
import matplotlib.font_manager as fm # 글꼴 관리
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  name='NanumGothic')
# ttf : truetype font
fm.fontManager.ttflist.insert(0, fe)
# rc : run control
plt.rcParams.update({'font.size': 14, 'font.family': 'NanumGothic'})
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# 이미지 사이즈 : 800(가로)x600(세로)
# 행열() 600x800 => 메모리 저장 ( 일렬로 저장 )
# pixel의 위치 3번째 행의 100번째열의 일렬의 위치는
# 0, 1, 2, 3
# 행우선으로 저장
def print_color_image(image_path):
    try:
        img = Image.open(image_path)
        img_array = np.array(img) # ndarray
        height, width, channels = img_array.shape
        print(f"이미지 크기: {width} x {height}, 채널 수: {channels}")
        print("이미지 데이터 (일부):")
        for y in range(min(10, height)): # 행
            for x in range(min(10, width)): # 열
                print(img_array[y, x], end=" ") # img_array[y, x,1]
            print()
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(img_array, cmap='Reds')
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(red_channel, cmap='Reds')
        plt.title("Red Channel")
        plt.subplot(132)
        plt.imshow(green_channel, cmap='Greens')
        plt.title("Green Channel")
        plt.subplot(133)
        plt.imshow(blue_channel, cmap='Blues')
        plt.title("Blue Channel")
        plt.show()

    except FileNotFoundError:
        print(f"오류: {image_path} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류: {e}")
image_path = "/content/drive/MyDrive/dataset/paduk.jpg"
print_color_image(image_path)

import matplotlib.pyplot as plt
from scipy import ndimage, datasets
face = datasets.face()
# 표준편차가 3
blurred_face = ndimage.gaussian_filter(face, sigma=3)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(face)
plt.title('Original Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(blurred_face)
plt.title('Blurred Image ($\sigma=3$)')
plt.axis('off')
plt.show()

# edge detection

from scipy import ndimage, datasets
import matplotlib.pyplot as plt
import numpy as np

face = datasets.face(gray=False)
# 흑백 이미지로 변환 R*0.299 + G*0.587 + B*0.114
gray_face = np.dot(face[..., :3], [0.299, 0.587, 0.114])

# 1. 수평 방향 (x축) 엣지 검출
sobel_x = ndimage.sobel(gray_face, axis=1) # 열방향

# 2. 수직 방향 (y축) 엣지 검출
sobel_y = ndimage.sobel(gray_face, axis=0) # 행방향

# 3. 전체 기울기 강도 (Magnitude) 계산 # 두 개 다 계산
# sqrt(X**2 + y**2)
sobel_magnitude = np.hypot(sobel_x, sobel_y)

fig, axes = plt.subplots(1, 4, figsize=(18, 6))

axes[0].imshow(gray_face, cmap=plt.cm.gray)
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(sobel_x, cmap=plt.cm.gray)
axes[1].set_title('Sobel X (Vertical Edges)')
axes[1].axis('off')

axes[2].imshow(sobel_y, cmap=plt.cm.gray)
axes[2].set_title('Sobel Y (Horizontal Edges)')
axes[2].axis('off')

axes[3].imshow(sobel_magnitude, cmap=plt.cm.gray)
axes[3].set_title('Sobel Magnitude (Total Edges)')
axes[3].axis('off')

plt.tight_layout()
plt.show()

# conv1d
from keras.models import Sequential # 순차형 모델
# 행렬곱, convolution1d, flatten 평탄화
# MaxPooling1D : 최대값
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
iris = load_iris()
# Bunch형 데이터
# 150 x 4
x, y = iris.data, iris.target
print(x.shape)
# 형태 일치
# 이미지는 4차원
# 배치사이즈 * 행 * 열 * channel
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # 150x4x1

print(unique(y)) # class수 3개
print(unique(y).sum())

# train : 0.85 / test : 0.15
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

model = Sequential()
model.add(Input(shape=(4,1))) # 하나의 데이터에 대해서 모델을 구성
# 64는 나가는 차수
# convolution 연산마다 relu activation function이 계산됨
# 주변값을 고려해서 특징 추출
# (4,1)
# 커널 차수 : 2
model.add(Conv1D(32, 2, activation="relu"))
model.add(MaxPooling1D())
# FFNN으로 전환
model.add(Flatten()) # 64
# 64 * 128 + 128
model.add(Dense(128, activation="relu"))
# 128 * 3 + 3
# output activation function : softmax
model.add(Dense(3, activation = 'softmax'))
# 원핫인코딩 되지 않은 실제값 과 softmax를 매핑해서 손실 함수
model.compile(loss = 'sparse_categorical_crossentropy',
     optimizer = "adam",
              metrics = ['accuracy'])
model.summary()
model.fit(xtrain, ytrain, batch_size=16,epochs=100, verbose=0)
# evaluate => loss, accuracy 자동 계산
acc = model.evaluate(xtest, ytest)
print("Loss:", acc[0], " Accuracy:", acc[1])

pred = model.predict(xtest) # softmax로 예측
pred_y = pred.argmax(axis=-1) # augmax로 라벨을 결정

cm = confusion_matrix(ytest, pred_y)
print(cm)

import numpy as np
import matplotlib.pyplot as plt

image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# kernel = filter
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

def convolve2d(img, filt):
    h, w = img.shape # 5x5
    kh, kw = filt.shape # 3x3
    # 출력 (3, 3)
    output = np.zeros((h - kh + 1, w - kw + 1))

    for i in range(output.shape[0]): # 3
        for j in range(output.shape[1]): # 3
            region = img[i:i+kh, j:j+kw] # 현재 위치가 (i, j) 필터 범위만큼
            output[i, j] = np.sum(region * filt)
    return output
feature_map = convolve2d(image, kernel)

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Kernel")
plt.imshow(kernel, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Feature Map (Convolution)")
plt.imshow(feature_map, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
# featuremap은 입력데이터와 커널. 즉, 필터를 적용한 결과

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# sequential, Module을 상속
# functional (input vs output)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist

(x_train, _), _ = mnist.load_data() # train 데이터만
img = x_train[0] # 첫 번째 이미지 한 장 28x28 => 784
img = img.astype('float32') / 255.0 # 이미지 정규화
# 차원확대
# CNN에서 이미지 포맷 (배치사이즈 * w * h * channel)
img = np.expand_dims(img, axis=(0, -1)) # 1x28x28x1

# 모델은 1개의 데이터
input_layer = Input(shape=(28, 28, 1))
# 필터 수 (kernel 수), 필터사이즈는 (3x3)
# same 나가는 차수가 입력되는 차수와 동일하게 충전재를 채워서
# padding : valid, same
# 필터 가중치 공간 (3x3) x 8
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1')(input_layer) # 28x28x8
x = MaxPooling2D((2, 2))(x) # 14x14x8
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x) # 14x14x16
x = MaxPooling2D((2, 2))(x) # 7x7x16
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x) # 7x7x32
model = Model(inputs=input_layer, outputs=x)
# feature map을 추출
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
# 데이터를 입력하면 featuremap을 출력하는 모델
feature_map_model = Model(inputs=model.input, outputs=layer_outputs)
feature_maps = feature_map_model.predict(img)

def plot_feature_maps(feature_maps, layer_names):
  for fmap, name in zip(feature_maps, layer_names):
    # 28x28x8
    num_filters = fmap.shape[-1] # 마지막 차원 => 필터 수
    size = fmap.shape[1]
    display_filters = min(num_filters, 8) # 최소한 8장만
    plt.figure(figsize=(display_filters * 2, 2))
    for i in range(display_filters): # 8장
      plt.subplot(1, display_filters, i + 1)
      # feature map 출력
      plt.imshow(fmap[0, :, :, i], cmap='viridis')
      plt.axis('off')
    plt.suptitle(f'Layer: {name} | Feature maps')
# 이름을 결정
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
plot_feature_maps(feature_maps, layer_names)

model.layers

# Mnist CNN Network

import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (_, _) = mnist.load_data()

train_images.shape # (60000, 28, 28)

# 차수 형식을 맞추기 위해서
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32') / 255

# CNN에서 가중치 필터 수 * 필터 사이즈 + 바이어스
3*3*32 + 32 # 320

3*3*64+64

model = tf.keras.Sequential([
    # valid : 28 - 3 + 1 = 26
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # 26x26x32
    tf.keras.layers.MaxPooling2D((2, 2)), # 13x13x32
    # 13 - 3 + 1 = 11
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # 11x11x64
    tf.keras.layers.MaxPooling2D((2, 2)), # 소수점 이하는 절사 : 5x5x64
    tf.keras.layers.Flatten(), # 1600
    tf.keras.layers.Dense(64, activation='relu'), # 가중치 사이즈 1600 x 64
    tf.keras.layers.Dense(10, activation='softmax') # 64 x 10
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

model.summary()

conv1_weights = model.layers[0].get_weights()[1]
tf.shape(conv1_weights)

conv1_weights = model.layers[2].get_weights()[1]
tf.shape(conv1_weights)

# 64개가 나갈 때 32장을 고려해서 64개를 만들어야 함
3 * 3 * 32 * 64 + 64 # 18496

# 학습이 끝난 상태의 가중치
# 가중치[0] + bias[1]
conv1_weights = model.layers[0].get_weights()[0] # conv2d
conv2_weights = model.layers[2].get_weights()[0] # conv2d

plt.figure(figsize=(12, 4))
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(conv1_weights[:, :, 0,i ], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('Convolutional Layer 1 Kernels', fontsize=16)
plt.figure(figsize=(12, 4))
for i in range(64):
    ax = plt.subplot(4, 16, i + 1)
    plt.imshow(conv2_weights[:, :, 0, i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('Convolutional Layer 2 Kernels', fontsize=16)
plt.show()

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape) # (60000, 28, 28) 채널이 없음

x_train = x_train[..., tf.newaxis] # 차수를 증가
x_test = x_test[..., tf.newaxis]
x_train.shape

# data 자동 feeding
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Model을 상속
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # 3x3, valid
    # 28x28x1 => 26x26x32
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten() # 21632
    self.d1 = Dense(128, activation='relu') # 21632 x 128
    self.d2 = Dense(10, activation='softmax') # 128 x 10
  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x) # softmax (10) 값이 리턴
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
# 전체 계산
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

@tf.function # static graph를 구성
def train_step(images, labels):
  with tf.GradientTape() as tape: # 미분 객체
    predictions = model(images)
    loss = loss_object(labels, predictions) # 실제 라벨 : 예측 라벨
  # 경사하강법으로 미분
  gradients = tape.gradient(loss, model.trainable_variables)
  # 가중치하고 바이어스 미분 결과를 적용
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images) # softmax값이 리턴
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
  for images, labels in train_ds: # 자동 feeding 해주는 객체
    train_step(images, labels)
  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)
  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))


import matplotlib.pyplot as plt
import numpy
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
# 26x26x32 -> 64
# 26x26x32x64
model.add(Conv2D(64, (3, 3), activation='relu')) # 24x24x64
model.add(MaxPooling2D(pool_size=2)) # 12x12x64
model.add(Dropout(0.25))
model.add(Flatten()) # 12x12x64 => 9216
model.add(Dense(128, activation='relu')) # 9216x128
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # 128x10
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

import os
MODEL_DIR = '/content/drive/MyDrive/model/cnnmodel/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

modelpath="{epoch:02d}-{val_loss:.4f}.keras"
modelpos = os.path.join(MODEL_DIR, modelpath)
# val_loss 개선되면 저장
checkpointer = ModelCheckpoint(filepath=modelpos, monitor='val_loss', verbose=1,
                               save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30,
                    batch_size=200, verbose=0, callbacks=[early_stopping_callback,
                                                          checkpointer])
# loss(0), accuracy(1)
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 문제
# 학습이 끝난 후 저장된 모델을 로딩해서 연속해서 학습하도록 구현하시오

from tensorflow.keras.models import load_model

# 저장된 모델 로드
model = load_model('/content/drive/MyDrive/model/cnnmodel/10-0.0262.keras')

# 학습 이어서 진행
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    initial_epoch=20,
    epochs=30,
    batch_size=200
)