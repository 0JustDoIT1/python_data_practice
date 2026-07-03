# ! pip show keras # Version : 3.13.2 -> 버전 맞추는거 중
import tensorflow as tf
print(f"설치된 TensorFlow 버전: {tf.__version__}") # 2.20.0

# FFNN(feed forward neral network)
# 전진망 : 행렬곱 연산
# 특징추출, 차원축소
# 행렬곱 연산은 내적연산의 연속이다.
# 내적 연산은 크기가 고려된 사이각
# 구성이 graph로 되어 있어서 CPU, GPU 독립적 실행이 가능
# 병렬처리가 가능
# tensorflow 2.x는 1.x와 다르게 eager mode를 지원
# eager mode는 dynamic graph


# GPU(Graphic process unit)
# 부동소수점 병렬연산
# nvidia
#  - cuda를 이용해서 연산 실행
#  - cudnn으로 딥러닝 학습 :

# 런타임을 T4 GPU로 변경
import tensorflow as tf
tf.test.gpu_device_name()

print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# tensorflow 2.x -> numpy, pandas 통합한 것은 문법
import tensorflow as tf
# tensorflow나 pytorch 다 변환 => tensor
scalar_constant = tf.constant(10) # 상수
vector_constant = tf.constant([1, 2, 3]) # vector
matrix_constant = tf.constant([[1,2], [3,4]], dtype = tf.float32) # matrix
shaped_constant = tf.constant(5, shape=[2,2])
scalar_constant, vector_constant, matrix_constant, shaped_constant
# 호환을 위해서 numpy로 변환해서 확인

C = tf.constant(10.0) # 상수
V = tf.Variable(5.0)  # 변수 - 가중치를 저장
try:
  C.assign(12.0)
except AttributeError as e: # 상수는 수정이 불가
  print(f"상수 업데이트 실패 (Constant): {e}")
V.assign(7.0)
V.assign_add(3.0) # 누적 할당
print(f"변수 업데이트 성공(Variable): {V.numpy()}")
V_old = V # 주소가 복사
V = tf.constant(99.0) # 다른값 할당
# tensor -> numpy롤 변환
# 결과값도 numpy로 변환
print(f"일반 할당 후 V_old 값: {V_old.numpy()} (값 유지됨)")

x = tf.constant([2.0, 1.0, 1.0, 4, 3, 4, -4, 2, 2], shape =[3,3])
y = tf.constant([1.0, 2, -6], shape = [3,1]) # 3x1
x_1 = tf.linalg.inv(x) # 역행렬
# 3x3  3x1  => 3x1
z = tf.matmul(x_1, y) # 행렬곱 연산
print(z)
# tf.Tensor
print(2.0 * z[0] + 1.0 * z[1]+ 1.0 *z[2])

# 연립방정식의 해를 solve
z = tf.linalg.solve(x, y)
print(2.0 * z[0] + 1.0 * z[1]+ 1.0 *z[2])

def gugu(dan):
  level = tf.constant(dan) # 상수
  state = tf.Variable(0)   # 가중치 변수
  for _ in range(9):       # 맨 마지막 연산의 결과값을 저장하는 _ (참조하지 않겠다)
    state.assign_add(1)
    result = level * state # 텐소도 일반 연산자를 사용
    # numpy 변환
    print(' {} x {} = {:2}'.format(level.numpy(), state.numpy(), result.numpy()))

for _ in range(2):
  gugu(_)

import numpy as np
numpy_array = np.array([5,6,7], dtype = np.int64)
tensor_b = tf. constant(numpy_array)
tensor_b = tf.convert_to_tensor(numpy_array)
print(f"Tensor Dtype: {tensor_b.dtype}")

initial_weights_np = np.random.randn(10,5).astype(np.float32)
# 가중치 저장 -> 가중치 초기화
weights = tf.Variable(initial_value = initial_weights_np)
print(f"Variable Type: {type(weights)}")
weights.numpy()

# 초기화
# sigmoid, tnah : GlorotNormal
# relu : HeNormal
# selu : LeCunNormal
# keras : backend: tensorflow, pytorch
# tensorflow가 keras하고 가까운 이유는 내부적으로 붙박이
he_initializer = tf.keras.initializers.HeNormal()
he_tensor = he_initializer(shape=[2,2])
print("He Tensor:", he_tensor)
weights = tf.Variable(initial_value = he_tensor)
print(f"Variable Type: {type(weights)}")
weights.numpy()


# 특이 행렬 분해
import tensorflow as tf
a = tf.constant([[1,2], [3,4], [5,6]], dtype = tf.float32) # 3 by 2
s, u, v = tf.linalg.svd(a)
print(s)
print(u)
print(v)

# 원래 값으로 복귀
# s가 고유벡터
a_reconstructed = u @ tf.linalg.diag(s) @ tf.transpose(v)
print(a_reconstructed)

# 행렬의 요소연산
print(a - a_reconstructed)
difference = a - a_reconstructed

epsilon = tf.keras.backend.epsilon()
epsilon # 1e-07 => 0.0000001 / 부동소수점은 비교가 불가능
# 엡실론 상수 = 기준보다 작으면 같다고 봄
# reduce : 여러개르 다 처리
is_equal = tf.reduce_all(tf.abs(difference) < epsilon)
is_equal

a = tf.constant([[1,2], [3,4]], dtype =tf.float32)
# norm은 벡터의 크기
norm_l1 = tf.linalg.norm(a, ord =1)
norm_l2 = tf.linalg.norm(a, ord =2)
print(norm_l1)
print(norm_l2)

# 문제 : 두 행렬의 행렬곱을 구하고 이를 출력하시오.
a = tf.constant([1,2,3,4,5,6], shape = [2,3])
b = tf.constant([7,8,9,10,11,12], shape = [3,2])
c = tf.matmul(a,b)
c = a @ b
print(c)
print(c.numpy())
c = tf.matmul(b,a) # 3x2  2x3  =>  3x3

# 문제2 : [1,2,3,4,5,6] 벡터를 [2,3] 상수행렬로 선언하고
# 거듭제곱한 값을 출력해 보시오.
x1 = tf.constant([1,2,3,4,5,6], shape =[2,3])
x2 = tf.transpose(x1) # 전치
result = tf.matmul(x1, x2)
result.numpy()


# 문제3: 다음을 대각행렬로 구성하시오.
diagonal = [1,2,3,4]
print(tf.linalg.diag(diagonal))

tf.linalg.diag_part(tf.linalg.diag(diagonal)).numpy()

# range, arnage(adarray), data_range(pandas의 index), period_range
# Tensorflow의 range -> Tensor
a = tf.range(6, dtype = tf.int32) # Tensor
print("a      :", a)
a_2d = tf.reshape(a, (2,3)) # 2차원
print("a_2d    :", a_2d)
a_3d = tf.expand_dims(a_2d, 0) # 3차원
print("a_3d    :", a_3d)
a_4d = tf.expand_dims(a_3d, 3) # 4차원
print("a_4d    :", a_4d)

a_1d = tf.squeeze(a_4d) # 복원
print("a_1d     :", a_1d)

# padding -> CNN할 때 원래차수를 지원
t = [[1,2,3], [4,5,6]]
        #    행      열
paddings = [[1,1], [2,2]]
tf.pad(t, paddings, "CONSTANT")

# reflect
paddings = [[1,1], [1,1]]
print(tf.pad(t, paddings, "REFLECT"))

# symmetric : 경계값을 포함해서 대칭 복사하여 패딩하는 방식
print(tf.pad(t, paddings, "SYMMETRIC")) # 대칭

# concat
t1 = [[1,2,3], [4,5,6]]
t2 = [[7,8,9], [10,11,12]]
print(tf.concat([t1, t2], 0).numpy()) # 행방향
print(tf.concat([t1, t2], 1).numpy()) # 열방향

# bias
X = tf.constant([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype = tf.float32)
print(X.numpy())
W = tf.constant([ # 가중치
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], dtype = tf.float32) # 행렬연산, 내적연산 후에는 bias가 붙음 (비선형으로 매핑하기 위해서)
# bias가 있는 이유 y = Wx + b(절편)
# bias의 개수는 나가는 차수와 일치
# 기울기가 0이되면 -> 가중치가 0으로 가면 학습이 중단
# 2x2 + b (가중치가 )이 되는 것을 방지, b가 높으면 민감도가 커짐
b = tf.constant([0.1, 0.2], dtype = tf.float32) # bias
Y_matmul = tf.matmul(X,W) + b
print(Y_matmul.numpy())

# 나가는 차수를 결정
# y = Wx + b
dense_layer = tf.keras.layers.Dense(units =2, use_bias = True)
Y_dense = dense_layer(X) # 2x3  가중치  3x2
print(Y_dense.numpy())
dense_W, dense_b = dense_layer.get_weights() # 공간이 자동으로 형성되니까 이름을 몰라서 get함수가 있음
print("\n가중치(W) 크기:", dense_W.shape) # 3x2 나가는 차수와 일치
print("편향 b 크기:", dense_b.shape)      # 2

# 학습
# y = as + b # a가중치
X = tf.constant([[1.0], [2.0], [3.0]], dtype = tf.float32)
y = tf.constant([[6.0], [7.0], [8.0]], dtype = tf.float32)
# 네트워크 망을 구성
dense_layer = tf.keras.layers.Dense(units =1, use_bias = True)
_ = dense_layer(X)
_, initial_bias = dense_layer.get_weights() # 바이어스만
print(f"값: {initial_bias} (기본값인 0으로 초기화되어 있음)") # 0
print(f"타입 확인: {dense_layer.bias.__class__}")
# SGD (stochastic gradient desecnt 확률적 경사 하강법 : 한번에 하나씩)
# 최적화기를 선택

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
# 자동 미분
# Tape : 이전 데이터 저장장치 (미분할 때 - 이전데이터가 필요)
with tf.GradientTape() as tape:
  y_pred = dense_layer(X) # 1개가 출력
  # MSE (mean squared error) : 차의 제곱의 평균
  # loss function  손실함수 -> 작게 학습
  loss = tf.reduce_mean(tf.square(y_pred - y))
  # 학습가능한 변수(가중치, 바이어스)
gradients = tape.gradient(loss, dense_layer.trainable_variables)
# 경사하강법 적용
optimizer.apply_gradients(zip(gradients, dense_layer.trainable_variables))
_, updated_bias = dense_layer.get_weights() # 가중치 확인
print(f"값: {updated_bias}")

# 미분
W = tf.Variable(5.0)
X = tf.constant(3.0)
with tf.GradientTape() as tape: # 미분객체
  y = W * X # y = 5x3
            # y = 3W -> 미분 -> 3
gradient = tape.gradient(y, W) # delta y / delta W 로 미분
# W로 편미분
print(f"변수 W의 값: {W.numpy()}")
print(f"계산된 기울기 (dy/dW): {gradient.numpy()}")

x = tf.constant(2.0) # 상수
with tf.GradientTape(persistent = True) as t: # 값을 유지
  t.watch(x) # 변수만 추적 -> 상수 추적
  y = 2*x    # -> 2
  y1 = x*x*x # x***3 -> 3x**2 => 12
dy_dx = t.gradient(y, x)
dy1_dx = t.gradient(y1, x)
print(dy_dx)
print(dy1_dx)

# 선형회귀
np.random.seed(42)
X_train = tf.convert_to_tensor(np.linspace(-1, 1, 100), dtype = tf.float32)
# y = a(2)x + b(1) + noise
y_train = tf.convert_to_tensor(2 * X_train.numpy() + 1 + np.random.normal(0, 0.1, size = 100), dtype = tf.float32)
w = tf.Variable(tf.random.normal([1]), name = "weight")
b = tf.Variable(tf.random.normal([1]), name = "bias")
learning_rate = 0.1
epochs = 50

for epoch in range(1, epochs + 1):
  with tf.GradientTape() as tape:
    y_pred = w * X_train + b # 가설 검정
    loss = tf.reduce_mean(tf.square(y_pred - y_train))
  # 가중치, 바이어스 역전파로 변경하기 위해
  dw, db = tape.gradient(loss, [w,b])
  w.assign_sub(learning_rate * dw) # 학습율 만큼만 적용
  b.assign_sub(learning_rate * db)
  if epoch % 10 == 0:
    print(f"Epoch {epoch:2d} -> Loss: {loss.numpy(): .4f} | w: {w.numpy()[0]: .4f} | b: {b.numpy()[0]: .4f}")
print(f"최종 가중치 w: {w.numpy()[0]: .4f}")
print(f"최종 편향   b: {b.numpy()[0]: .4f}")

# keras 형태 - 순차모델(layer가 순차적으로 실행 모델)
# C++처럼 구조를 만듬
model = tf.keras.Sequential([
    # 1x1 사이즈의 가중치
    tf.keras.layers.Dense(units = 1, input_shape = [1])
])

# keras는 추상화된 구조 -> backend 코드로 컴파일
model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1), loss= "mse")
model.fit(X_train, y_train, epochs = 50, verbose = 0)
weights, biases = model.layers[0].get_weights() # 가중치 체크
print(f"예측 결과: 기울기(w) = {weights[0][0]: .4f}, 절편(b) = {biases[0]: .4f}")
X_test = np.array([[2.0]])
y_pred = model.predict(X_test, verbose = 0)
print(f"\n 예측 y: {y_pred[0][0]: .4f} (정답 근사치: 5.0)")

# eager mode , graph mode

# eager mode , graph mode
# 명령하나에 GPU 실행이 한번
a = tf.constant(5)
b = tf.constant(3)
c = a + b
print(f"연산 결과: {c.numpy()}")
if c.numpy() > 5:
  print("조건 만족: 연산 결과가 5보다 큽니다.")

d = tf.Variable(2.0)
d.assign_add(1.5)
print(f"변수 값: {d.numpy()}")

# tensorflow 2.x 버전에서 eager mode 를 도입
# @tf.function를 이용해서 static mode를 살림
# pytorch 가 compile을 만들어서 static graph 방식의 장점을 가져옴
import time
# @tf.funciton이 없으면 실시간으로 dynamic graph를 생성
# @tf.function이 있으면 static graph를 만들도 cache에 저장해서 실행속도를 높임
@tf.function # graph decorator
def graph_add(x,y):
  return x + y

# 1차 때는 시간이 걸림
start_graph_1 = time.time()
result_graph_1 = graph_add(tf.constant(10), tf.constant(5))
end_graph_1 = time.time()
print(f"그래프 1차 호출 결과: {result_graph_1.numpy()} (시간: {end_graph_1 - start_graph_1: .4f}s)")

# 2차 때는 1차때 생성한 그라프를 사용 -> 실행 시간이 단축
start_graph_2 = time.time()
result_graph_2 = graph_add(tf.constant(20), tf.constant(10))
end_graph_2 = time.time()
print(f"그래프 2차 호출 결과: {result_graph_2.numpy()} (시간: {end_graph_2 - start_graph_2 : .4f}s)")

# SparseCategoricalCrossentropy
# 독립변수 / 종속변수([0,2] 원핫인코딩이 안된 상태)
import tensorflow as tf
y_true = [0,2] # 3개의 class (점수 클래스)
# onehotencoding은 안하더라도 차원을 일치시켜줘야 함
y_true = tf.constant([[0], [2]], dtype = tf.int32)
# softmax를 통과하면 확률값으로 출력
y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]],
                     dtype = tf.float32)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
loss = loss_fn(y_true, y_pred)
print("Sparse Loss:", loss.numpy())

y_true_one_hot = [[1,0,0], [0,0,1]]
y_pred = [[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]]

loss_fn = tf.keras.losses.CategoricalCrossentropy()
loss = loss_fn(y_true_one_hot, y_pred)
print("Categorical Loss:", loss.numpy())

# MNIST 데이터 모델
import tensorflow as tf
import numpy as np
import time
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 손글씨 데이터
print(x_train.shape) # (60000, 28, 28)

# fully-connected
# Deep learning -> 정규화
# 이미지는 정수형 (0-255 : 1바이트) RGB 3바이트로 표햔
# 255 이미지 정규화
x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0

# 정수형 (0~9 class)
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")

# validation 검증용 - 학습하면서 검증
# train -> validation -> test 3단계로 데이터 분리
x_val = x_train[-10000:]
y_val = y_train[-10000:]    #10000

x_train = x_train[:-10000]  # 50000
y_train = y_train[:-10000]  # 50000

BATCH_SIZE = 64 # 한번에 GPU에 전달되는 데이터 사이즈
BUFFER_SIZE = 1024

# RGB 28x28
3*28*28*64  # 150528 -> GPU 메모리

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 데이터 전달 준비중

# data feeding
# 64개씩 분할해서 전달
# ft.data.Dataset
# 메모리 사정에 따라서 미리 로딩하는 양을 조절하면서
# chaining 체이닝을 지원
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size = BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (x_val, y_val)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# tf.Module 클래스를 상속받아서 모델을 정의
class DenseModel(tf.Module):
  def __init__(self, input_size = 784, hidden_units = 64, num_classes=10):
    super().__init__()
    # sigmoid, tanh activation function을 사용할 때
    initializer = tf.initializers.GlorotUniform()
    # 가중치  784 x 64 : 차원을 축소하면서 특징추출
    self.w1 = tf.Variable(initializer(shape=[input_size, hidden_units]), name = "w1")
    # bias를 0으로 초기화
    self.b1 = tf.Variable(tf.zeros(shape=[hidden_units]), name ="b1")
    # 행렬곱이 순서적으로 실행
    # 64 x 64
    self.w2 = tf.Variable(initializer(shape=[hidden_units, hidden_units]), name = "w2")
    self.b2 = tf.Variable(tf.zeros(shape=[hidden_units]), name ="b2")
    self.w_out = tf.Variable(initializer(
        shape = [hidden_units, num_classes]), name="w_out")
    self.b_out = tf.Variable(tf.zeros(shape=[num_classes]), name ="b_out")

  @tf.function
  def __call__(self, x): # 입력되는 학습 데이터
    h1 = tf.matmul(x, self.w1) + self.b1
    h1 = tf.nn.relu(h1) # 비선형 매칭
    h2 = tf.matmul(h1, self.w2) + self.b2
    h2 = tf.nn.relu(h2) # 값을 제한
    logits = tf.matmul(h2, self.w_out) + self.b_out
    return logits # 행렬곱을 이용한 연산 마지막 출력

model = DenseModel()

# 최적화기 손실함수를 정의

NUM_EPOCHS = 30 # 전체 데이터 학습
LEARNING_RATE = 1e-3  # 0.001
# 실제 데이터는 정수형 vs  예측 데이터는 class만큼 확률
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
# 최적화기 : 확률적 경사하강법
optimizer = tf.optimizers.SGD(learning_rate = LEARNING_RATE)
# 정수인덱스 : softmax 확률값을 이용해서 accuracy를 계산
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = "train_accuracy")
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = "val_accuracy")
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = "test_accuracy")
test_loss_metric = tf.keras.metrics.Mean(name = "test_loss")

# 실행함수 적용

@tf.function # static graph를 생성
def train_step(x, y): # 학습
  with tf.GradientTape()as tape:
    logits = model(x)
    # SparseCategoricalCrossentropy(정수형변수, 확률전값)
    loss_value = loss_fn(y, logits)
    # trainable_valiables 가중치 , Bias(3단계)
  grads = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_acc_metric.update_state(y, logits)
  return loss

@tf.function
def validate_step(x, y): # 검증
  val_logits = model(x)  # x 입력은 검증용 데이터 (10000)
  val_acc_metric.update_state(y, val_logits)

@tf.function
def test_step(x, y):
  test_logits = model(x) # 테스트 데이터 입력
  test_loss_metric.update_state(loss_fn(y, test_logits))
  test_acc_metric.update_state(y, test_logits)

# 실제 학습과 평가

# 실제 데이터를 feeding 하면서 학습
# validation -> 1 epoch마다 1번씩 실행
for epoch in range(NUM_EPOCHS): # 30
  start_time = time.time()
  # 64 x 784
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    train_loss = train_step(x_batch_train, y_batch_train)
    if step % 700 == 0:
      print(f"Epoch {epoch + 1} | Step {step} - " f"Loss: {train_loss.numpy():.4f} | Acc: {train_acc_metric.result().numpy():.4f}")

  for x_batch_val, y_batch_val in val_dataset:
    validate_step(x_batch_val, y_batch_val)
  print(f"\nEpoch {epoch + 1} Time {time.time() - start_time: .2f}s")
  print(f"Training Accuracy: {train_acc_metric.result().numpy():.4f}, Validation Accuracy: {val_acc_metric.result().numpy():.4f}")

# 학습이 완료되면 test 데이터 평가
for x_batch_test, y_batch_test in test_dataset: # 64장
  test_step(x_batch_test, y_batch_test)
print(f"Test Loss: {test_loss_metric.result().numpy():.4f}")
print(f"Test Accuracy: {test_acc_metric.result().numpy():.4f}")