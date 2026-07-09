# upsampling
import numpy as np
import tensorflow as tf
input_shape = (2,2,1,3) # 12
x = np.arange(np.prod(input_shape)).reshape(input_shape)
print(x)

# 확장
y = tf.keras.layers.UpSampling2D(size = (2,2))(x)
print(y.shape)
# CNN 모델 4차원
# 배치사이즈 행과 열 그리고 channel
# pooling할 때도 채널사이즈는 변동이 없음

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
input_data = np.random.rand(1,4,4,3).astype(np.float32)
print("Input Data Sahep:", input_data.shape)
model = models.Sequential([
    # channel output 5
    # 2배 사이즈로 확장
    # (4,4,3) 채널을 합해서 계산
    # 채널별로 계산 = DepthwiseConv2D
    # pointwise : 1x1로 convolution : Conv2D(kernel_size = 1) : 채널을 하나로 묶음

    layers.Conv2DTranspose(5, kernel_size = 3, strides = 2, activation = "relu",
                           padding = "same", input_shape=(4,4,3)),
])
model.summary()

# autoencoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import matplotlib.pyplot as plt
import numpy as np

# _ : 정답 라벨인데, 오토인코더는 정답 숫자 라벨을 사용하지 않음
(X_train, _), (X_test, _) = mnist.load_data()
# 이미지 형태 변환
# 원래 shape: (60000, 28, 28)
# CNN에 넣기 위해 shape: (60000, 28, 28, 1) 로 변경
# 1은 흑백 이미지라는 뜻
# /255를 해서 픽셀값을 0~1 사이로 정규화
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32") / 255
autoencoder = Sequential()

# 28x28x16
autoencoder.add(Conv2D(16, kernel_size = 3, padding = "same", input_shape=(28,28,1), activation = "relu"))
# 14x14x16
autoencoder.add(MaxPooling2D(pool_size =2, padding = "same"))
# 14x14x8
autoencoder.add(Conv2D(8, kernel_size =3, activation = "relu", padding = "same"))
# 7x7x8
autoencoder.add(MaxPooling2D(pool_size=2, padding = "same"))
# 4x4x8 -> strides = 2도 사이즈를 반으로 줄
autoencoder.add(Conv2D(8, kernel_size = 3, strides = 2, padding = "same", activation = "relu"))
# 4x4x8
autoencoder.add(Conv2D(8, kernel_size =3, padding = "same", activation = "relu"))
# 8x8x8
autoencoder.add(UpSampling2D())
# 8x8x8
autoencoder.add(Conv2D(8, kernel_size =3, padding = "same", activation = "relu"))
# 16x16x8
autoencoder.add(UpSampling2D())
# 14x14x16 - padding="same"이 없어서 한개가 줄어듬
autoencoder.add(Conv2D(16, kernel_size = 3, activation = "relu"))
# 28x28x16
autoencoder.add(UpSampling2D())
# 28x28x1
autoencoder.add(Conv2D(1, kernel_size =3, padding = "same", activation = "sigmoid"))
# linear -> MSE
# sigmoid -> binary_crossentropy (현재 해당사항)

autoencoder.summary()

autoencoder.compile(optimizer = "adam", loss = "binary_crossentropy")
autoencoder.fit(X_train, X_train, epochs = 50, batch_size = 128, validation_data = (X_test, X_test))

# 원본이미지 vs acutoencoder가 생성한 이미지
# 시각화
random_test = np.random.randint(X_test.shape[0], size = 5)
ae_imgs = autoencoder.predict(X_test)
plt.figure(figsize = (7,2))
for i, image_idx in enumerate(random_test):
  ax = plt.subplot(2,7, i+1)
  plt.imshow(X_test[image_idx].reshape(28,28))
  ax.axis("off")
  ax = plt.subplot(2,7,7+i+1)
  plt.imshow(ae_imgs[image_idx].reshape(28,28))
  ax.axis("off")
plt.show()

# autoencoder의 denoising

# encoder /특징/ decoder
# encoder -> 차원축소된 특징추출(noise가 사라짐)
def utility_denoising():
  noise_factor = 0.4
  X_test_noisy = X_test + noise_factor * np.random.normal(
      loc = 0.0, scale = 1.0, size = X_test.shape)
  X_test_noisy = np.clip(X_test_noisy, 0., 1.)
  denoised_imgs = autoencoder.predict(X_test_noisy, verbose = 0)
  plt.figure(figsize =(6,3))
  for i in range(4):
    plt.subplot(2,4,i+1)
    plt.imshow(X_test_noisy[i].squeeze(), cmap="gray")
    plt.axis("off")
    if i ==0: plt.title("Noisy Input", fontsize = 9)
    plt.subplot(2,4,i+5)
    plt.imshow(denoised_imgs[i].squeeze(), cmap="gray")
    plt.axis("off")
    if i ==0: plt.title("Denoised Output", fontsize =9)
  plt.suptitle("Denoising", y=1.02, fontweight = "bold")
utility_denoising()

# 이상탐지

# 이상탐지 (정상데이터에 대비하여)
# encoder + decoder
# 원본이미지 -> noise 제거 -> 특징이 추출 -> 복원(노이지가 없어진 데이터로)
# 정상이미지 : AE에서 학습된 이미지 내의 것들
# 이상이미지는 복원시 차이가 많이 발
def utility_anomaly_detection():
  anomaly_data = np.random.uniform(0,1,(10,28,28,1))
  # 복원된 이미지
  normal_recon = autoencoder.predict(X_test[:10], verbose = 0)
  # 비정상 데이터
  anomaly_recon = autoencoder.predict(anomaly_data, verbose=0)
  normal_mse = np.mean(np.square(X_test[:10] - normal_recon), axis=(1,2,3))
  anomaly_mse = np.mean(np.square(anomaly_data - anomaly_recon), axis=(1,2,3))
  print(f"정상 평균 복원 오차: {np.mean(normal_mse): .5f}")
  print(f"이상 평균 복원 오차: {np.mean(anomaly_mse): .5f}")
utility_anomaly_detection()
# 많은 데이터를 확인 : threshold를 결정

encoder_input = Input(shape=(28, 28, 1))
x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(encoder_input)
x = MaxPooling2D(pool_size=2, padding='same')(x) # 14x14
x = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=2, padding='same')(x) # 7x7
latent_features = Flatten(name='latent_space')(x)
encoder_model = Model(encoder_input, latent_features, name="Encoder")

decoder_input = Input(shape=(392,))
x_dec = Reshape((7, 7, 8))(decoder_input)
x_dec = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x_dec)
x_dec = UpSampling2D()(x_dec) # 14x14
x_dec = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x_dec)
x_dec = UpSampling2D()(x_dec) # 28x28
x_dec = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x_dec)
decoder_output = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x_dec)
decoder_model = Model(decoder_input, decoder_output, name="Decoder")
ae_output = decoder_model(encoder_model(encoder_input))
autoencoder = Model(encoder_input, ae_output, name="Autoencoder")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=5, batch_size=128, validation_data=(X_test, X_test))

# 이미지 유사도에 의한 검색
# 이미지 -> AE를 이용한 특징 추출 -> 저장
# encoder model 이용
# 입력되는 이미지를 잠재벡터 추출 -> KNN을 이용해서 Ball Tree를 만들어 놓고 검색 or 근접검색
def utility_image_retrieval(query_idx=0):
  # 잠재 벡터 -> 인코더 모델로 테스트 이미지들의 압축 특징 추출
  latent_vectors = encoder_model.predict(X_test, verbose= 0)
  # 검색할 벡터 -> 기준 이미지 하나 선택
  # 입력되는 이미지를 잠재벡터 추
  query_vector = latent_vectors[query_idx]
  # 거리값 = 모든 데이터의 잠재벡터 - 질의 벡터 ( 거리가 작을수록 비슷한 이미지)
  distances = np.linalg.norm(latent_vectors - query_vector, axis=1)
  # 비슷한 이미지 검색 = 가장 가까운 이미지 4개 선택 ([0]은 자기 자신이므로 제외)
  closest_indices = np.argsort(distances)[1:5]
  # 검색된 데이터 출력
  plt.figure(figsize=(6,2))
  plt.subplot(1,5,1)
  plt.imshow(X_test[query_idx].squeeze(), cmap = "gray")
  plt.axis("off")

  # 비슷한 이미지 4개
  for i, idx in enumerate(closest_indices):
    plt.subplot(1,5,i+2)
    plt.imshow(X_test[idx].squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"Match {i+1}", fontsize = 9)
  plt.suptitle("Similarity Search", y= 1.05, fontweight = "bold")
  plt.tight_layout()
  plt.show()
utility_image_retrieval()

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import tensorflow as tf

(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print(x_train.shape)
print(x_test.shape)

latent_dim = 64
class Autoencoder(Model):
  def __init__(self, encoding_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(latent_dim, activation = "relu")
    ])
    self.decoder = tf.keras.Sequential([
        layers.Dense(784, activation = "sigmoid"),
        layers.Reshape((28,28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer = "adam", loss = losses.MeanSquaredError())

autoencoder.fit(x_train, x_train, epochs = 10,
                shuffle = True,
                validation_data =(x_test, x_test))

# 64바이트로 표현된 이미지
# 783 -> 64 특징 추출
encoded_imgs = autoencoder.encoder(x_test).numpy()
print(encoded_imgs.shape)
decoded_imgs = autoencoder(x_test).numpy()
decoded_imgs.shape

# original(원본) vs reconstruct(재구성)
n = 10
plt.figure(figsize = (20,4))
for i in range(n):
  ax = plt.subplot(2,n, i+1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax = plt.subplot(2,n,i+1+n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

# FFNN망을 이용한 분류모델
# 지도학습이라 y_train 포함
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
NUM_CLASSES = 10
feature_extractor = autoencoder.encoder # 가중치는 그대로
classifier_model = Sequential([
    feature_extractor, # 레이어에 추가 (특징 추출):64
    # 64 x 128
    layers.Dense(128, activation = "relu", name ="classifier_dense"),
    layers.Dropout(0.3),
    # 128 x 10
    layers.Dense(NUM_CLASSES, activation = "softmax",
                 name = "output_layer")
], name = "AE_Classifier")
classifier_model.compile(
    optimizer = "adam",
    loss = losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

history_cls = classifier_model.fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 256,
    shuffle = True,
    validation_data = (x_test, y_test),
    verbose = 1
)
loss, acc = classifier_model.evaluate(x_test, y_test, verbose = 0)
print(f"AE 특징기반 분류모델 테스트 정확도: {acc:.4f}")

# RandomForest
# 신경망으로 특징추출 -> SVM으로 학습한것과 같은 원리
# 딥러닝으로 특징 추출 후 -> randomforest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# 딥러닝과 일치시키기 위해서 정규화
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# 28x28x1로 구성하기 위해서
if len(x_train.shape) == 3: # AE로 특징추출
  # encoder만 사용해서 추출
  x_train = np.expand_dims(x_train, axis = -1)
  x_test = np.expand_dims(x_test, axis = -1)
feature_extractor = autoencoder.encoder
x_train_features = feature_extractor.predict(x_train, batch_size = 256, verbose = 0)
x_test_features = feature_extractor.predict(x_test, batch_size = 256, verbose = 0)

if len(x_train_features.shape) > 2:
  x_train_features = x_train_features.reshape(
      x_train_features.shape[0], -1) # 2차원으로 축소
  x_test_features = x_test_features.reshape(
      x_test_features.shape[0], -1)
rf_classifier = RandomForestClassifier(
    n_estimators = 100,
    max_depth = 15,
    random_state = 42,
    n_jobs = -1,
    verbose =1
)

rf_classifier.fit(x_train_features, y_train.squeeze())
y_pred = rf_classifier.predict(x_test_features)
rf_acc = accuracy_score(y_test.squeeze(), y_pred)
print(f"AE 특징기반 RF 분류모델 테스트 정확도: {rf_acc:.4f}")

# GAN

# GAN 이미지 생성모델
# mnist학습 -> 숫자를 noise에 의해서 자동생성

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
if not os.path.exists("/content/drive/MyDrive/dataset_gan/gan_images"):
  os.makedirs("/content/drive/MyDrive/dataset_gan/gan_images")

# noise를 입력으로 받아서 28x28x1 사이즈의 이미지를 생성하는 함수
def build_generator(latent_dim): # 잠재벡터의 크기를 입력받음
  model = keras.Sequential(name = "generator")
  model.add(layers.Input(shape=(latent_dim,)))
  # 나가는 차수는 모양을 만들기 위해서
  # 28x28x1 사이즈의 이미지
  model.add(layers.Dense(7*7*256, use_bias = False))
  model.add(layers.BatchNormalization()) # 배치사이즈마다 정규화(평균이 0)
  model.add(layers.LeakyReLU(0.2)) # ReLU의 단점: -를 고려하지 않음 -> -1까지만 고려된 개선사항(LeakyReLU)
  model.add(layers.Reshape((7,7,256)))
  # 14x14x128
  # stride : conv2d에서는 줄이는 역할 / Conv2DTranspose에서는 늘리는 역할
  model.add(layers.Conv2DTranspose(128, 4, strides =2,
                                  padding = "same", use_bias = False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(0.2))
  # 28x28x64
  model.add(layers.Conv2DTranspose(64,4,strides =2,
                                   padding = "same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(0.2))
  # 28x28x1
  model.add(layers.Conv2D(1,7,padding="same",
                          activation = "tanh"))
  return model

# 판별자(생성한 이미지와 진짜이미지를 구별)
# img_shape = (28,28,1) -> 원래 이미지 사이즈
def build_discriminator(img_shape):
  model = keras.Sequential(name = "discriminator")
  model.add(layers.Input(shape = img_shape)) # 28x28x1
  # 14x14x64
  model.add(layers.Conv2D(64,4, strides =2, padding = "same"))
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Dropout(0.3))
  # 7x7x128
  model.add(layers.Conv2D(128,4,strides = 2, padding = "same"))
  model.add(layers.LeakyReLU(0.2))
  model.add(layers.Dropout(0.3))
  model.add(layers.Flatten()) # 7x7x128 => 6272
  model.add(layers.Dense(1)) # 6272 x 1
  return model

def load_data(): # 비지도학습 - 이미지 학습
  (x_train, _), (_,_) = keras.datasets.mnist.load_data()
  x_train = x_train.astype("float32")
  x_train = (x_train - 127.5) / 127.5 # -1 ~ 1 : 학습효과가 큼
  x_train = np.expand_dims(x_train, axis =-1) # 차원증가

  dataset = tf.data.Dataset.from_tensor_slices(x_train)
  dataset = dataset.shuffle(60000).batch(128,
                                         drop_remainder= True).prefetch(tf.data.AUTOTUNE) # 메모리사정 고려해서
  return dataset

latent_dim = 100 # noise사이즈
img_shape = (28,28,1) # 생성하는 이미지 사이즈, 진짜 이미지 사이즈
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape) # 판별자 경찰관
# 손실함수
bce = keras.losses.BinaryCrossentropy(from_logits=True)
# momentum을 조절 => 0.9 기준학습 : 너무 확신하지 말것
g_optimizer = keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5)
d_optimizer = keras.optimizers.Adam(learning_rate= 0.0002, beta_1 = 0.5)

# 손실함수
def discriminator_loss(real_output, fake_output):
  # 진짜 이미지는 항상 참 : 1.0: 과신 : label smoothing
  real_loss = bce(tf.ones_like(real_output) * 0.9, real_output)
  # 생성된 이미지는 항상 0으로 봄
  fake_loss = bce(tf.zeros_like(fake_output), fake_output)
  return real_loss + fake_loss

def generator_loss(fake_output): # 생성자 : 판별자를 속일 수 있도록 학습
  # 항상 자기가 생성한 것이 참
  return bce(tf.ones_like(fake_output), fake_output)

# 실제 이미지 : 가짜이미지는 noise로부터 생성
@tf.function
def train_step(real_images):
  batch_size = tf.shape(real_images)[0] # 기본 32, 100
  noise = tf.random.normal([batch_size, latent_dim])
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    # 가짜 이미지 생성
    fake_images = generator(noise, training = True)
    # 결과값은 진짜 이미지에 대한 판별값
    real_output = discriminator(real_images, training = True)
    # 가짜 이미지에 대한 판별값 0.01 ~ 점점 상승하는 방향으로 상승
    fake_output = discriminator(fake_images, training = True)
    d_loss = discriminator_loss(real_output, fake_output)
    g_loss = generator_loss(fake_output) # 자기가 확신하는 1: 실제 생성한 이미지
  d_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables) # 가중치 조
  g_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)

  d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
  g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
  return d_loss, g_loss

def show_images(epoch, examples = 25):
  noise = np.random.normal(0,1, (examples, latent_dim))
  generated = generator(noise, training = False)
  # BatchNormalization 정규분
  # 이미지 -1 ~ 1 사이값 => 이미지는 양수만 가능해서 +1 해줌 => 0~2사이값을 가짐
  # 부동소수점 이미지 0 ~ 1
  generated = (generated + 1) / 2.0
  fig = plt.figure(figsize = (5,5))
  for i in range(examples):
    plt.subplot(5,5,i+1)
    plt.imshow(generated[i, :, :, 0], cmap = "gray")
  plt.suptitle(f"Epoch {epoch}")
  plt.tight_layout()
  fig.savefig("/content/drive/MyDrive/dataset_gan/gan_images/gan_mnist_%d.png")
  plt.show()

dataset = load_data()
epochs = 50
for epoch in range(1, epochs +1):
  d_losses = []
  g_losses = []
  for real_images in dataset:
    d_loss, g_loss = train_step(real_images)
    d_losses.append(d_loss)
    g_losses.append(g_loss)
  print(
      f"Epoch {epoch:03d} |"
      f"D loss: {tf.reduce_mean(d_losses): .4f} |"
      f"G loss: {tf.reduce_mean(g_losses): .4f}"
  )
  if epoch % 5 ==0 or epoch ==1:
    show_images(epoch)

# 모델을 저장하고 로딩한 다음 노이즈를 받아서
# 이미지를 생성한 다음 추력하는 함수를 작성하시오
# 저장할때는 h5 => keras로 저장(반드시 확장자 지정)
# 노이즈 입력

import os
# !pip install gdown -qq
FILE_ID = '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS'
OUTPUT_FILENAME = 'celeb_a_sub_data.zip'
TEMP_DIR = '/tmp'
OUTPUT_PATH = os.path.join(TEMP_DIR, OUTPUT_FILENAME)
# !gdown --id "$FILE_ID" -O "$OUTPUT_PATH"
# !unzip -q "$OUTPUT_PATH" -d "$TEMP_DIR/celeb_a_data"

# !pip install --upgrade --force-reinstall tensorflow-datasets
# 세션 다시 시작하라는 알람




import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
IMG_SIZE = 64
BATCH_SIZE = 64
def preprocess(example):
    image = tf.image.resize(example['image'], (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, image
ds = tfds.load("celeb_a", split='train', shuffle_files=True)
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


from datasets import load_dataset
import tensorflow as tf
IMG_SIZE = 64
BATCH_SIZE = 64
ds_hf = load_dataset("nielsr/CelebA-faces", split="train")
def preprocess_hf(example):
    image = example["image"]
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, image

ds = tf.data.Dataset.from_generator(
    lambda: (preprocess_hf(x) for x in ds_hf),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
    )
)
ds = ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

from tensorflow.keras import layers, models
def build_encoder():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),# 64x64x3
        # filter 4x4
        layers.Conv2D(32, 4, strides=2, padding='same', activation='relu'),# 32x32x32
        layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),# 16x16x64
        layers.Conv2D(128, 4, strides=2, padding='same', activation='relu'),# 8x8x128
        layers.Flatten(),# 8192
        layers.Dense(256, activation='relu') # 256
    ])
    return model
def build_decoder():
    model = models.Sequential([
        layers.Input(shape=(256,)),
        layers.Dense(8 * 8 * 128, activation='relu'),# 8192
        layers.Reshape((8, 8, 128)), # 8x8x128
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'), # 16x16x64
        layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),# 32x32x32
        layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')# 64x64x3 ( 노이즈 제거 )
    ])
    return model
encoder = build_encoder()
decoder = build_decoder()
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
z = encoder(inputs)
outputs = decoder(z)
autoencoder = tf.keras.Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

EPOCHS = 10
autoencoder.fit(ds, epochs=EPOCHS)

# 잠재벡터를 이용해서 특성을 변화해 이미지를 생성

# 잠재벡터 : encoder로 부터 압축된 백터
# 잠재벡터 : 벡터를 랜덤한 방향으로 (어떻게 이미지가 바뀔지 모)
latent_dim = encoder.output_shape[1:]
attribute_axis = np.random.rand(*latent_dim).astype(np.float32)
# 정규
attribute_axis = attribute_axis / np.linalg.norm(attribute_axis)
# 원본이미지 3장
original_images = next(iter(ds))[0][:3]
plt.figure(figsize=(15,6))
for i, img in enumerate(original_images):
  plt.subplot(2,3,i+1)
  plt.imshow(img)
  plt.title(f"Original {i+1}")
  plt.axis("off")
  # 잠재벡터
  latent_vec = encoder.predict(np.expand_dims(img, axis=0), verbose =0)
  alpha = 5.0
  # 잠재벡터 + 임의 방향으로 강조점(특징강조)
  edited_latent_vec = latent_vec + alpha * attribute_axis
  # decoder를 이용해서 생성 : 강조한 방향으로 이미지가 재생성
  edited_img = decoder.predict(edited_latent_vec, verbose = 0)[0]
  plt.subplot(2,3,i+4)
  plt.imshow(edited_img)
  plt.title(f"Edited {i+1} (+Att)")
  plt.axis("off")
plt.suptitle("Attribute edited")
plt.tight_layout()
plt.show()
# 웃는 얼굴을 만들고 싶다면
# 웃는 얼굴을 인코더에 넣고 잠재벡터
# 무표정한 얼굴을 인터에 넣고 잠재벡터
# 이 잠재벡터의 차이가 곧 웃는 얼굴의 특성 벡터
# 입력이미지의 잠재벡터 + 웃는 얼굴의 잠재벅터를 더하면 => 원래 이미지가 웃는 얼굴로 변

# 과제
# GAN을 구성할때 함수베이스로 수업시간에 작업한 것을
# 이를 클래스 베이스로 변환해서 작업해보시오
# 파일 제출