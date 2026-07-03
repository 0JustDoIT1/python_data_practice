import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
inputs = Input(shape=(3,), name='input_vector')
# 가중치 사이즈 : 3x2 + bias
outputs = Dense(2, use_bias=True, name='dense_linear')(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
x_sample = np.array([[1.0, 2.0, 3.0]])
y_pred = model.predict(x_sample)
print("입력:", x_sample)
print("출력:", y_pred)
# 입력가중치는 없음 -> 데이터만 입력
# 8인 이유 = 3x2 + 2 = 8

model.get_config()
# trainable : True => 가중치 학습을 한다
# false => 가중치 학습을 안하는 경우 (그 레이어가 학습이 잘 된 경우)
# dense activation function default : linear
# dense initializer : GlorotUniform
# bias_initializer : Zero

for layer in model.layers: # 순서적으로 지정
  print("레이어이름:", layer.name)
weights = model.get_weights() # 가중치
print("입력:", model.inputs)
print("출력:", model.outputs)
config = model.get_config()
print("총 파라미터 수:", model.count_params()) # 총 파라미터 개수

from tensorflow.keras import layers
from tensorflow import keras
model = keras.Sequential([
    layers.Input(shape=(3,)), # 열이 3개
    # 3x2
    layers.Dense(2, activation="relu", name="layer1"),
    # 2x3
    layers.Dense(3, activation="relu", name="layer2"),
    # 3x4
    layers.Dense(4, name="layer3"),
])
x = tf.ones((3, 3)) # 데이터 개수 x 열 수
y = model(x) # 3x4
y

model.layers

model.layers[0].weights # 3x2

# %load_ext tensorboard

from keras.datasets import mnist
import matplotlib.pylab as plt
(X_train0, y_train0), (X_test0, y_test0) = mnist.load_data()
print(X_train0.shape, X_train0.dtype) # 60000 x 28 x 28
print(X_test0.shape, X_test0.dtype) # 10000
print(y_test0.shape, y_test0.dtype)
plt.imshow(X_train0[0]) # 28x28 => 784
plt.grid(False)
plt.show()
# FFNN ( feed forward neural network ) => 입력데이터 ( fully-connected )

# X_train0 : 60000x28x28
# 신경망 : 정규화
# 이미지 정규화 : 255로 나눔
# 부동소수점 변환
X_train = X_train0.reshape(60000, 784).astype('float32')/255.0
X_test = X_test0.reshape(10000, 784).astype('float32')/255.0
print(X_train.shape, X_train.dtype)

y_train0[:10] #
# sparcecategorticalcrossentropy

from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(y_train0, 10)
Y_test = to_categorical(y_test0, 10)
Y_train[:5]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras.optimizers import SGD
import numpy as np
import datetime
np.random.seed(0)
model = Sequential() # 인스턴스 후 add 함수로 레이어 추가
# 100 x 784
# 1개를 기준 네트워크를 생성
model.add(Input(shape=(784,))) # 100 x 784
model.add(Dense(15, activation="sigmoid")) # 784 x 15 => 100x15
model.add(Dense(10, activation="softmax")) # 15x10 => 100x10
# optimizer, loss, metrics
model.compile(optimizer=SGD(learning_rate=0.2),
              # loss='mean_squared_error', # 회귀
              loss='categorical_crossentropy',
              metrics=["accuracy"])
# data, epoches, batch_size
hist = model.fit(X_train, Y_train, epochs=30, batch_size=100,
                 validation_data=(X_test, Y_test), verbose=2)

import matplotlib.pylab as plt
plt.plot(hist.history['loss'])
plt.show()

# epoch마다 loss, accuracy 값을 자동 저장
plt.plot(hist.history['accuracy'])
plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return Sequential([
      # 28x28 => 784
      tf.keras.layers.Flatten(input_shape=(28,28)),
      # 784x512
      Dense(512, activation="relu"),
      # 과적합방지, 계산회로 중에 무작위로 20%를 삭제
      # ensemble 학습효과
      tf.keras.layers.Dropout(0.2), # trabinable 의 20%
      # 512x10
      Dense(10, activation="softmax")
  ])

model = create_model()
model.compile(optimizer='adam', # momentum을 제공, learning_rate 조절
              loss='sparse_categorical_crossentropy', # 정수 index : softmax
              metrics=['accuracy'])
log_dir = "logs/fit/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1 # 1 epoch 마다 1번씩
)
hist = model.fit(
    x=x_train,
    y=y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback]
)

model.summary()
# 784 * 512 + 512
# 512 * 10 + 10

for layer in model.layers:
  print(layer.name)

l1 = model.layers[1]
l1.name

l1.activation

# %load_ext tensorboard

# %tensorboard --logdir logs/fit

# !curl -L -o insurance.zip "https://www.kaggle.com/api/v1/datasets/download/mirichoi0218/insurance"

# !unzip insurance.zip -d /content/drive/MyDrive/dataset_medical/insurance/

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('/content/drive/MyDrive/dataset_medical/insurance/insurance.csv')
except FileNotFoundError:
    print("파일 없음")

X = df.drop('charges', axis=1)
y = df['charges'] # 종속변수 charges

df.head()

numerical_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]
preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder = "passthrough"
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
if hasattr(X_train_processed, 'toarray'):
  X_train_processed = X_train_processed.toarray()
  X_test_processed = X_test_processed.toarray()
input_dim = X_train_processed.shape[1]
print(f"FFNN 입력 차원 (특징 수): {input_dim}")

def ffnn_regression_model(input_dim):
  model = keras.Sequential([
      layers.Input(shape=(input_dim,)),
      layers.Dense(units=128, activation='relu', name='dense_1'),
      layers.Dropout(0.2),
      layers.Dense(units=64, activation='relu', name='dense_2'),
      layers.Dense(units=1, activation=None, name='output_layer')
  ], name="Medical_Cost")
  model.compile(
      optimizer = keras.optimizers.Adam(learning_rate=0.001),
      loss='mse',
      metrics=['mae']
  )
  return model

ffnn_model = ffnn_regression_model(input_dim)
ffnn_model.summary()

EPOCHS = 100
BATCH_SIZE = 32
history = ffnn_model.fit(
    X_train_processed,
    y_train,
    validation_data=(X_test_processed, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

loss, mae = ffnn_model.evaluate(X_test_processed, y_test, verbose=0)
y_pred = ffnn_model.predict(X_test_processed).flatten()
print(f"(MSE Loss): {loss:,.2f}")
print(f"(MAE - $ 예측 정확도): {mae:,.2f} $")

# 모델 개선 - 규제 추가
from tensorflow import keras

def create_improved_ffnn_model(input_dim):
  l2_reg_strength = 0.001
  model = keras.Sequential([
      layers.Input(shape=(input_dim,)),
      layers.Dense(units=256, activation='relu',
                   kernel_regularizer=regularizers.l2(
                       l2_reg_strength)),
      layers.BatchNormalization(),
      layers.Dropout(0.3),
      layers.Dense(units=128, activation='relu',
                   kernel_regularizer=regularizers.l2(
                       l2_reg_strength)),
      layers.BatchNormalization(),
      layers.Dropout(0.3),
      layers.Dense(units=64, activation='relu',
                   kernel_regularizer=regularizers.l2(
                       l2_reg_strength)),
      layers.Dense(units=1, activation=None)
  ], name = "Improved_Predictor")
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.0005),
      loss='mse',
      metrics=['mae']
  )
  return model

checkpoint_path = '/content/drive/MyDrive/dataset_medical/insurance_model/model.keras'

callbacks = [
    keras.callbacks.EarlyStopping( # 과적합방지
        monitor='val_mae',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau( # 개선이 없을 때 LR 값을 줄임
        monitor='val_mae',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
]

input_dim = 11
improved_model = create_improved_ffnn_model(input_dim)

history = improved_model.fit(
    X_train_processed,
    y_train,
    validation_data = (X_test_processed, y_test),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=0
)
loss, mae = improved_model.evaluate(X_test_processed, y_test, verbose=0)
print(f"새로운 평균 절대 오차 (MAE): {mae:,.2f} $")

# 문제
# activation function 을 ReLU, GeLU, Mish로 비교해서 MAE 차이를 분석하시오
# 개선방안을 찾아 보시오 (1가지씩)
# 파생변수
# 전처리 : 범주형변수 one-hot-encoding 적용
# region 변수 제거 고려

# 모델 개선 - 규제 추가
from tensorflow import keras

def create_improved_ffnn_model(input_dim):
  l2_reg_strength = 0.001
  model = keras.Sequential([
      layers.Input(shape=(input_dim,)),
      layers.Dense(units=256, activation='gelu',
                   kernel_regularizer=regularizers.l2(
                       l2_reg_strength)),
      layers.BatchNormalization(),
      layers.Dropout(0.3),
      layers.Dense(units=128, activation='gelu',
                   kernel_regularizer=regularizers.l2(
                       l2_reg_strength)),
      layers.BatchNormalization(),
      layers.Dropout(0.3),
      layers.Dense(units=64, activation='gelu',
                   kernel_regularizer=regularizers.l2(
                       l2_reg_strength)),
      layers.Dense(units=1, activation=None)
  ], name = "Improved_Predictor")
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.0005),
      loss='mse',
      metrics=['mae']
  )
  return model

# optuna
# !pip install optuna

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import KFold
import optuna
import numpy as np

X = np.random.rand(100, 13)
y = np.random.rand(100, 1)

# parameter tunning을 전제로 구성된 모델
# optuna는 가장 좋을 것으로 기대되는 확률이 높은 조합으로 테스트
def create_model(trial): # 시도횟수조절 객체
  # 변동 parameter
  learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
  # 가던 방향을 고려해서 전진
  momentum = trial.suggest_float('momentum', 0.8, 0.999, log=True)
  # hidden layer 나가는 차수
  units_1 = trial.suggest_int('units_1', 10, 50)
  units_2 = trial.suggest_int('units_2', 5, 20)
  model = Sequential()
  model.add(Input(shape=(13,))) # 변수 13
  # 13x10 ~ 13x50
  model.add(Dense(units_1, kernel_initializer='normal', activation='relu')) # gelu, mish
  model.add(Dense(units_2, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # adam + nestronov ( 끝단으로 이동한 다음 거기서 방향 )
  # E( exponent 지수 ) MA( moving average 이동평균법 )
  opt = keras.optimizers.Nadam(learning_rate=learning_rate, use_ema=True, ema_momentum=momentum)
  model.compile(loss='mean_squared_error', optimizer=opt)
  return model


# 목적함수 - 최적화의 대상
def objective(trial, X, y):
  # cross validation : CV
  n_splits = 3
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
  mse_scores=[]
  # 3번에 걸쳐 feeding()
  for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model = create_model(trial)
    model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)
    if trial.should_prune(): # 가지치기
      raise optuna.exceptions.TrialPruned() # 예외 발생
    mse = model.evaluate(X_val, y_val, verbose=0)
    mse_scores.append(mse)
  return np.mean(mse_scores)

def tune_hyperparameters(X, y):
  # 가지치기 : 모델 평가 : 이미 완료된 평가 모델의 평균값
  # 실시간으로 파라미터 확률을 다시 계산
  pruner = optuna.pruners.MedianPruner(
      n_startup_trials=5,
      n_warmup_steps=30,
      interval_steps=1
  )
  study = optuna.create_study(direction='minimize', pruner=pruner)
  func = lambda trial: objective(trial, X, y)
  study.optimize(func, n_trials=100)
  print("시도횟수: ", len(study.trials))
  print("최적시도: ")
  trial = study.best_trial
  print(" Value: {}".format(trial.value))
  print(" Params: ")
  for key, value in trial.params.items():
    print(" {}: {}".format(key, value))
  return study, study.best_params
study, best_params = tune_hyperparameters(X, y)
print(best_params)

from sklearn.model_selection import train_test_split
def create_final_model(params, input_dim=13):
  model = Sequential()
  model.add(Input(shape=(input_dim,)))
  model.add(Dense(params['units_1'],
                  kernel_initializer='normal', activation='relu'))
  model.add(Dense(params['units_2'],
                  kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  opt = keras.optimizers.Nadam(learning_rate=params['learning_rate'])
  model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', 'mse'])
  return model
final_model = create_final_model(best_params, input_dim=13)
X_full_train, X_final_test, y_full_train, y_final_test = train_test_split(X, y, test_size=0.2, random_state=42)

import os
from tensorflow.keras.callbacks import ModelCheckpoint
FINAL_EPOCHS = 120
FINAL_BATCH_SIZE = 5
CHECKPOINT_PATH = 'best_model/weights'
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_PATH, 'best_weights.keras'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

history = final_model.fit(
    X_full_train, y_full_train,
    epochs = FINAL_EPOCHS,
    batch_size=FINAL_BATCH_SIZE,
    validation_data = (X_final_test, y_final_test),
    verbose=1
)
loss, mae, mse = final_model.evaluate(X_final_test, y_final_test, verbose=0)
print(f"Loss (MSE): {loss:.6f}")
print(f"MAE: {mae:.6f}")

import optuna.visualization as vis
def visualize_optuna_study(study):
  fig_history = vis.plot_optimization_history(study)
  fig_history.show()
  fig_importance = vis.plot_param_importances(study)
  fig_importance.show()
  fig_slice = vis.plot_slice(study)
  fig_slice.show()

visualize_optuna_study(study)

# optuna도 early-stopping

# 팀과제:
# 1) kaggle에서 의료 tabla 데이터( 프로젝트 주제2 )를 다운로드하시오
# 2) FFNN 모델을 구축 ( 분류 문제 )
# 3) optuna를 이용해서 parameter tunning을 실시한 다음 최적의 파라미터로 모델을 학습시키시오
# 4) 결과를 tensorboard를 이용해서 시각화하시오