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

plt.rcParams['axes.unicode_minus'] = False

# artificial neural network (인공신경망)

import numpy as np
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

def MLP(x, w, b):
  y = np.sum(w * x) + b
  if y <= 0: # 기준값 0 비트 연산
    return 0
  else:
    return 1

# MLP 회로의 가중치와 바이어스를 조절해서 하나 통합
# 반도체 NAND 회로로 다 구성함
def NAND(x1, x2):
  return MLP(np.array([x1, x2]), w11, b1) # [-2, 2], 3
# OR
# 0   0   ->  0   ->  -1   ---> 0
# 0   1   ->  1   ->  1    ---> 1
# 1   0   ->  1   ->  1    ---> 1
# 1   1   ->  1   ->  3    ---> 1
def OR(x1, x2):
  return MLP(np.array([x1, x2]), w12, b2) # [2, 2], -1
def AND(x1, x2):
  return MLP(np.array([x1, x2]), w2, b3) # [1, 1], -1
# xor은 MLP 단일 칩으로 구현이 불가능
def XOR(x1, x2):
  return AND(NAND(x1, x2), OR(x1, x2))

print("NAND 문제")
for x in [(0, 0), (1, 0), (0,1), (1, 1)]:
  y = NAND(x[0], x[1])
  print("입력 값: "+ str(x) + "출력 값: " + str(y))

# ![image.png](data:image/png;base64,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)

# 반가산기 : XOR 회로 1개, AND 회로 1개로 구성
def Half_Adder(x1, x2):
  S = XOR(x1, x2)
  C = AND(x1, x2)
  return S, C
# 0 0 ->
# 1 0 -> 숫자가 다르면 1
# 0 1 -> 숫자가 다르면 1
# 1 1 -> 올림 발생

for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
  sum_result, carry_result = Half_Adder(x[0], x[1])
  print(f" 입력: {x[0]} + {x[1]} -> [자리올림수(Carry): {carry_result}] [합(Sum): {sum_result}]")

# 전가산기는 이전의 올림수를 고려해서 연산
# 전가산기는 반가산기 2개에 OR회로로 구성
def FullAdder(A, B, Cin):
  # 반가산기
  sum1 = XOR(A, B) # 1, 0 나 0, 1인 경우
  carry1 = AND(A, B) # A, B가 둘 다 1인 경우는 올림수 발생
  # 반가산기
  Sum = XOR(sum1, Cin)
  carry2 = AND(sum1, Cin) # 올림수 발생
  # OR 회로
  Cout = OR(carry1, carry2) # 둘 중에 하나만 1이어도 올림수 발생
  return Sum, Cout

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 시각화
def plot_decision_boundary(model, X, y, ax, title):
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                       np.arange(y_min, y_max, 0.01))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
  ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
  ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=80)
  ax.set_title(title)
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  ax.set_xticks([0, 1])
  ax.set_yticks([0, 1])
  ax.grid(True, linestyle='--')

# 신경망 분류기
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 회로구성은 1나의 데이터 처리를 기준 구성
# 배치 사이즈 고려
model_nonlinear = MLPClassifier( # multi layer perceptron
    hidden_layer_sizes=(2,), # 입력데이터 (4x2) 가중치 (2x2) 출력가중치 (2x1)
    # 분류 true/false -> sigmoid가 output 활성화 함수
    activation='tanh', # 값의 범위 -1 ~ 1
    solver='sgd', # light 버전 (준 뉴톤법 - 소량의 데이터)
    max_iter=1000, # 1회, 데이터를 늘리는 방법
    random_state=42,
    # early_stopping=True # when solver=’sgd’ or ‘adam’
)
model_nonlinear.fit(X_scaled, y)
acc_nonlinear = model_nonlinear.score(X_scaled, y)
# The test_size = 1 should be greater or equal to the number of classes = 2
# 테스트 데이터가 작아서 적용할 수 없음
# validation_fraction : 테스트 데이터가 작다
# best_validation_score_ : early_stopping 에 대한 결과값

model_nonlinear2 = MLPClassifier( # multi layer perceptron
    hidden_layer_sizes=(4,), # 입력데이터 (4x2) 가중치 (2x4) 출력가중치 (4x3), (3x1)
    # 분류 true/false -> sigmoid가 output 활성화 함수
    activation='tanh', # 값의 범위 -1 ~ 1
    solver='lbfgs', # light 버전 (준 뉴톤법 - 소량의 데이터)
    max_iter=1000, # 1회, 데이터를 늘리는 방법
    random_state=42
)
model_nonlinear2.fit(X_scaled, y)
acc_nonlinear2 = model_nonlinear2.score(X_scaled, y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(
    model_nonlinear, X_scaled, y, axes[0],
    f"은닉층1개(비선형)\nAccuracy: {acc_nonlinear:.2f}"
)

plot_decision_boundary(
    model_nonlinear2, X_scaled, y, axes[1],
    f"은닉층2개(비선형)\nAccuracy: {acc_nonlinear2:.2f}"
)

import numpy as np
# 소수점 이하를 비교하기 위해서
# 소수점 0.1222 222332 비교가 안됨
def binary_cross_entropy(y_true, y_pred):
  epsilon = 1e-15 # log(0) -> -무한대 로 가는 것을 방지
  # clip : 데이터의 범위를 제한할 때
  # 예측값
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  # 1,    0
  # 0.2,  0.4
  # 실제 데이터에 대한 확률값
  # log 확률이 높으면 손실은 작아야 됨, 확률이 작으면 손실은 커야 됨
  loss = -np.mean(y_true * np.log(y_pred) + # 참일 때
   (1 - y_true) * np.log(1 - y_pred)) # 거짓일 때
  return loss
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.9, 0.4, 0.3, 0.85, 0.05])
loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy Loss (Log-Loss): {loss:.4f}")

# 결과값을 확률값으로 변환
# 신경망이 출력하는 것은 logit 값
def softmax(z): # exp(지수함수)를 사용 : 큰 값은 크게, 작은 값은 작게 (차이를 극대화)
  exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
  return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 비용 함수
# logloss와 같은 의미
def categorical_cross_entropy(y_true, y_pred):
  epsilon = 1e-15
  y_pred = np.clip(y_pred, epsilon, 1-epsilon)
  # 확률값이 높으면 손실은 작아져야 하니까
  loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
  return loss

# 실제값
y_true = np.array([
    [1, 0, 0], # 클래스 3개가 원핫인코딩
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])

logits = np.array([
    [2.0, 1.0, 0.1],  # 2 / 3.1
    [1.2, 2.5, 0.3],
    [0.3, 0.8, 2.0],
    [2.1, 1.1, 0.2],
    [1.3, 2.0, 0.5]
])  

y_pred = softmax(logits) # 확률값으로 변환

# 1, 0, 0
# 실제 true인 값에 대한 loss를 계산하게 됨
loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy Loss: {loss:.4f}")

# SGDRegressor는 SGD를 적용해서 만든 회귀
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler() # 수치 안정
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 신경망(행렬곱 연산으로 구성) - 비선형(activation function)
mlp_reg = MLPRegressor(
    # 4개의 레이어
    hidden_layer_sizes=(100, 100), # 1000x1 -> 1x100 -> 100x100, 100x1 # 예측은 한 개의 데이터로 출력
    activation='relu', # sigmoid, tanh, relu (0 이하는 0으로 나머지는 그대로 : 학습속도가 빨라짐 - 데이터(특성 수) 절감)
    solver='adam', # lbfgs, sgd, adam(learning-rate adaptive 하게 조절(beta_1(학습률), beta_2(보폭을 조절)), momentum)
    max_iter=1000, # epoch(전체데이터학습 1회) 회수
    random_state=42,
    learning_rate_init=0.01, # 학습율 초기값
    n_iter_no_change=20, # 회수가 지났는데 개선 변화(loss)가 없으면
    verbose=False, # 설명 출력
    early_stopping=True # n_iter_no_change(참을 횟수), validation_fraction 테스트 비율
)
mlp_reg.fit(X_train_scaled, y_train)

y_pred_mlp = mlp_reg.predict(X_test_scaled)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print(f"MLP Regressor MSE: {mse_mlp:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
X_test_plot = np.sort(X_test, axis=0)
y_pred_plot = mlp_reg.predict(scaler.transform(X_test_plot))
plt.plot(X_test_plot, y_pred_plot, color='red', linewidth=2, label='MLP Prediction')
plt.xlabel("X")
plt.ylabel("y")
plt.title("MLP Non-linear")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.preprocessing import PolynomialFeatures

# 비선형 데이터 구성
np.random.seed(42)
X = np.random.uniform(-3, 3, 1000).reshape(-1, 1)
y = X**3 - 3*X**2 + 2*X + np.random.normal(0, 3, X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
degree = 3
poly_sgd = make_pipeline(PolynomialFeatures(degree),
                         StandardScaler(), SGDRegressor(max_iter=1000,
                                                        eta0=0.01,
                                                        learning_rate='constant',
                                                        penalty=None,
                                                        random_state=42))
poly_sgd.fit(X_train, y_train)
y_pred = poly_sgd.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"SGD Regressor (Polynomial) MSE: {mse:.4f}")

# 문제 (자체적으로 비선형 가능)
# MLPRegressor로 변환해 보시오
mlp_reg = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(20, 10),
        activation='relu',
        solver='adam',
        max_iter=3000, # 데이터 학습
        learning_rate_init=0.01,
        random_state=42
    )
)
mlp_reg.fit(X_train, y_train)
y_pred = mlp_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MLP Regressor MSE: {mse:.4f}")

# ![image.png](data:image/png;base64,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)

# sigmoid function 시각화
from matplotlib import pyplot
from math import exp
def sigmoid(x):
  return 1.0 / (1.0 + exp(-x))
inputs = [x for x in range(-10, 10)]
outputs = [sigmoid(x) for x in inputs]
pyplot.plot(inputs, outputs)
pyplot.show()

# ![image.png](data:image/png;base64,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)

# sigmoid의 결과가 좋지 않은 이유가 -가 없기 때문
# -1 ~ 1 -> tanh(모델 개선)

# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWUAAAEWCAIAAAAxURGDAAAQAElEQVR4AeydC3xVxbX/RUVU5BEFUVoe1lhsxZtWirQaBbyCClLxgVfkimgSjEIMSilgKKE8JFJF3oGQ3I+xCla5PhBLEa6AgMjj8qfcCBQEJDyCvMMFLM/8v91zuz/Hc05O9pyzz5zZc/Z8FsPas2fWrN+avdeeWbOzz4VVfvIt4FvAt4AzC1x4gZ98C/gW8C3gzAK+v3BmJ7+WbwHfAhdc4PsL/yrwLaDOAl7vyfcXXh9BX3/fAuos4PsLdbb2e/It4HUL+P7C6yMYvf5vvPHGhf9MtWvX/uEPf/jUU0/t3bvXicQlS5bQ9P333w+tTHlOTk5Q+Zw5cyj//PPPg8r9Q29ZwPcX3hov97UdNWrUW2+9NWPGjPvuuw+mffv2p06dirGbWrVqxSihuuZ+eWIt4PuLxNo/8b3jJh5//PGnn3565syZv/nNb7Zt2zZ37tzEq+VroKUFfH+h5bAkSKn09HR63r59Ozm0efPmRx555Kqrrrrsssvatm378ccfU+hTMlvA9xfJPPrB2L/55huKUlJSyL/66qtf/vKXf/vb34YOHTp+/Pi6det27979ww8/5JRPSWsB318k19CHoj169OjBgwd37979n//5n7///e8vvfTS+++/n2q5ubktW7Zct24di5Rnn32WAOevfvWrwYMHc8qnpLWA7y+Sduj/D/jdd9999dVXN2/evEePHldccQXBi6ZNmx4+fHjx4sWUVFZW4k0Ede7ceevWrRUVFf/X0v8v+Szg+4vkG/PvI542bdqiRYvY7+zSpQt+4ZJLLuH8119/XVVV9bvf/Q5XYtOIESNq1aq1f/9+KkRHNI+uod9KEwv4/kKTgUiYGrfeeutdd9310EMPMbNo3bo1eyUnTpw4f/48Cg0aNAhXEkgLFy68/vrrORWB6tSp89133wVVOHnyJCUsdsh98q4FfH8hN3YG177wwgvHjh27d+/eKVOmCKdw8cUX40qCiDVLZCO0aNGCjZWgOsRNKeEUuU/etYDvL7w7du5r3r59e6YbEyZMqF+/focOHWbMmLFv377Abg4cOBB4GJZnXfPll18SKLXPElJ9++23f/7zn7O0sQt9xosWiNVfHDt2jCusXr16GzduBP/y5ctvv/32O+64o6ysjEOfPGcB1iDffvvtG2+8MXXqVEIYN99880svvTRz5szRo0d37dq1U6dOgYiIelAeSOyzDBky5JprrrnzzjsHDhxYVFRE1ONnP/sZMtmUDWzr8160QKz+4vLLL//zn//8yCOPCPDDhg3jcNasWf7GmzCI5nloAJJABouR11577cYbb1y7di0+At/Rv39/7nyWJ/n5+QKRaPinP/1peEDibHl5OZOIVatWEQd59913aYjfSUtL+/zzz5m8iLZ+7l0LxOovuIYaNWok8BPluuiiixo0aNCsWTM25ERhUO4f6mOBPn36nDt37pZbbglUCUfApumWLVtgrrvuOpwFEY1Tp07hCD766KMHH3xQVObmJyZKc3KbOLztttuowI4s/mXXrl2nT59mCUPDtm3bUu6T1y0Qq78IxH/kyBHWvaIEP3L27FnB+7lvAd8CZljATX+RkpJCOEPYBWeByxA8sQxB4tDPfQv4FtDBAuKuJHeujGv+gtjYZZddhpuorKxkInrllVcKJdCGDTnWsc8+++w999zz61//mpwQOszdd9/drVs3wZDff//9nTt3hrnPSjAcsn6GoSY5JBiaI4RDcngYyhMuql27dsy6UUYrrbAP+rioFSuO1NRUxCJTB7OjibsAwYVAxIK0TZs2Cb+u0AR9bK1gJMx+993Uh2hCHiQqOzubTStC2kSsxN1aY+6Cv0CJTz/9NCsrq7S0lFA5h8S6CgoK7L5/8pOfLFmyZNmyZQsWLJg7dy45MVGYRYsWffzxx4IhnzdvHnJg5lsJhsNPPvkEhprkkGBojhAOyeFhKP/44wSLmj59Oot2lNFKK+yDPi5qRfwyIyMDscjUwexo4i5AcCEQsSAtLi7++OMEX1dogj62VjBumR2/zwOe+5SAI7kTcsFfgGfPnj1ffPHFk08+yU7qihUrcA04Lbt7Imc2bzBT20oGAxTQQElUW/Bm5yCFTMXIw5gHPOhqfAGPOja54C9sWWGZ1q1b24GMsBWMKaxnJWPgVAcElEnyWjdIoers4Ony48eP9+3bV0AYNmyYvU0hSiLkcfcX9H355ZeTG08trJQMMBs3bmw8TAD+YzxbtIAxj/ARO3fuBFfHjh1/+9vfwjgkFf6CUKhDbTxdTbyG4GkITpRPEpiYwlSkX3755aRJkwDIPJGIm9TtqcJffBfy14roah7ttpJ5uIIQgTJJXsYDKRQE3+uHp06dIlwtfMSoUaMIeUphVOEvkmQ90txKXr+eatQflPYbvTVW9nQFkEKehhCq/NixY8WferFVPGDAACpIYVThL8SeDZqZTSesZDZG0IHy73//O4wj8nIlkEJeRhCse1lZ2csvv0wpWxAlJSXk8FIYVfiLc+fOoZbxxLILSgaYp0+fNh4mABlNCMYM4jbMzMw8c+YMcAYPHpyWlgYDSWFU4S/q1KmDWsYTs3QoGWA6337ztDUYTcjTEAKVnzx58qpVqyhp1aoV+yMwgqQwqvAXhFiEZmbnB6xkNkbQgdL+KyEODSaQQmYA3LFjR15ensBSXFzMzojgyaUwqvAXSfI6YF0rMQBukn6yQJkkE0aQQvqNgLRG7Ib07dtXfEK1X79+4lepbClSGFX4CxFWsfUzlWEbCDIVnY0LjEniL0AK2cC9y5SWli5atAj9mzVrxv4ITCBJYVThL4RjC1TRSH6nlYyEFggKlFIz2MC23uJBCnlL51Bt9+3b9+KLL4ry6dOnh77hLoVRhb+QcmACmBdzg18fDhwOYPrvgwcaRHM+JyfnyJEjKNmrV68uXbrABBEDGlQS4VCFv4jQfdxP+R34FkhiC3z44Ydz5szBAGyCTJgwASZGUuEv/PVIjIOkVXOmr/56RKsRqU6Zo0ePPvfcc+LsxIkTcRmCD8oZ0KCSCIcq/IW/HokwAJ47xfTVX494YtQGDRokfuy2a9euPXv2rE5nBrS6U6HlKvzFueR4v/OUlUJNbFgJKMU7gobhCoUDUii03BMlixcvLi4uRlUCnIWFhRG+WSWFsSZ/QYcxU5JcXoetFLO1dBcAyuPHj+uupRv6gRRyQ5JqGUQAsrKyRK8FBQVsowo+bC6FUYW/CHyZLKzGZhReayUzsERAAcqUlJQIFYw5BVLIi3Dy8/O3bduG5unp6dnZ2TARSAqjCn/h+vxCzwUO4SUowsDEfkoH4GCU+ovG2FEnSgJIoUT1HnW/a9euFT89WadOHZYkF15Ywz0uhbEGWVErHb+GY8eOXbNmTY3yX3rppU2bNtVYzcUKLBEhFwUGiXIIPKhVLIdhbQhGKBaxXmkLTEittrH2xrM5IyPj/PnzCBo+fHirVq1gIpMURhX+IvJHlo8dO3bvvfe2bNkSR0h+3333de7c+Y477rjzzjunTp0qkNuAKWFb6Je//KVdUh2Tl5fHZtK+ffuqq+B6eQMruS5WCAwCXlZW9uCDDxIAf/7553NycqT+JFkIdJKHtSEok2TDC6SQE0PpU2fcuHEbNmxAn7S0NC4PmBpJCqMKfxH58yr169f/y1/+MnToUIDNmjVr/vz5n3766bJly0pKSn7/+993796dckFbtmyZM2eOHcgRhdXldevWxXZPPfVUdRVcL99rJdfFIjAI+JEjR+6+++6nn376D3/4w6RJk3ikRNgto3nUFNaGoESBqGV6qCFIIQ8pvHnz5pEjR6Iwj15un8jPaaoJksKowl9ccsklQrMI+fLly9n4CZw43HDDDR07dpw3b57wl7TFp/Tr1w/GIbVt25a9ooULFzqsH2O1xlaKUUjY5kHA8YOXXXZZt27dRGVsMnfu3BUrVohDd/NQG4KSkXK3Fz2lgRTSU7dQrZiJZ2Zmik8ZDRw4sE2bNqF1wpZIYVThL/B2YRUNLGRCwRokqObOnTtZXIn50jfffLN69eqHHnoosFWNPHN17q4aq7lSAXcOuSIqUEgocCZZt1k/gy6qtW7dmhv4vffeE4eu50E2BKPGf3DsJnqQQm5KjKeswsJC8cxITU0dMWKE866kMKrwF+wGR9Z+9+7d5eXl//qv/xpYbd26dcQ1mXWL98/efffdUIcSWD8sTxBk6dKlUjvMYeU4KcS7QU5qStUJAn78+HG2ygK/0YpLveaaazCXlFjnlYNsCEb/fXDn1lNTk9tnyJAhoq+ZM2dKBZgYUNHQSa7CX9SoPZMLdL3rrrvIBS1ZsuSxxx5jHl5UVCRKWFbceuutgg/KX3nllYYNG7Kk//bbb1nME7OwvzB81VVXsa4hIBLUJIrDN954o3///g8//DBTvtmzZzM8zPpYMW3dupWo6u9+97vp06czIfzjH/8YJPzjjz8Gy29/+9sBAwaMGTOGVoEVVq5c+cILL/AMxzP+7//+LzEqUMOzkhLVgoAz3aCcoA+5TQQa9u/fbx9GwTi3Ie5bagYbhTKaNAEppIkyEdSoqqrKzs7mQUIdonsdOnSAcU5SGFX4C/BE1h5/cemll77//vsEOF988cXrrrtu8uTJzAtGjx7Nw1O0Xb9+/U033ST4wPyDDz6gJrfZz372s9/85jc333wzN23ghPmnP/2pHQEJbNirV6+f15TwQaJJZWUlQccpU6Z8/vnnbEyws11QUPDaa68RR3j22WfRNj8/H19A1wwYPku0IiceiZfBlbAseuCBB5go/j3g49p/+9vfWEe8/vrrSNi+fTsjDQR8Ab5JfPQdCUHA2U6iMCgkhL+Q2kVHQiBJ2ZB1MhTY3FQemJD+6MQuAXo2bdqUywxGiqQwqvAXNe724S+Y9OIsuOvGjx+P41iwYEHgBBvfeejQISYRoYa48cYb//znP7Pz9+qrr/7oRz/C0d5+++2B1VJSUnbs2BFYIvi33377/9WUxFeJqP9f//VfbPriNVja/OIXv7DDKDimXbt2oTYMqyoqMPsQtzSt/vu//xv3xxRJaP4v//Iv586da9++PacE4U3E9905xEq4ryZNmqD/qFGj4CkMBS4+bmi7UepAeCh7SsWhLAXYsGYbAhMjyHbhxfoghTTX/MCBA7m5uULJadOmiStNHDrMpTCq8BeR1yNszvEsDdwZ4Vb5yU9+wgPcBsx9CB/WFtTkFPTSSy/x1GXODx9ItBLNAwtl+Xbt2nEb49eYK7F2sJuvXbu2T58+9EsJMQUc0w9+8AMWQRxCw4YN4/Cee+6Bh7744gvyQH8xePBgJlYUMulgEsTqBh7XifuDgYTmQIAXFHYtcOLEiSuuuEJUiCKXsiEwGzVqFEUvnmsCUkhztbngeZSiZI8ePZjAwsiSFEYV/iLyo4+gLjdhoL8AMPGbgwcPwggSj9PqJk40x8VyEdu3mWglcvZcIMFHnXPb82An7TOf1QAAEABJREFUqsKtRUdCDqsJIhf2/c9Ny8T+/vvvF2fxg4QeOnfuLA7JWcv8+Mc/ZgYBL8geKqIYBCwI6IpyOw8FfvXVV1OIcLsODF1LrUJpEkTObUhfeLeg5kYeghTSGdonn3zCYgQNmUSzpIWJgqQwqvAXTMIjwOChzdlAf8F0g1lW4H0ltlSDbhJaCSJAQNiCmb84FM9kwZPjfbnHYIKod+/ebWtK9tRAtCVQYnsHSjgkfnHrP6Ow7OYQ4yAswikmGl9//TUO7le/+hWHgkAqmodGpBcvXozvgETNHf9cQIUCZx6RlpaGPxU1yVmMMKUkfAMfNTm3Iesm1lxRd+ShhiCFtFWYZS+rb6EeIbDA+0UUOsylMKrwF0QHI6jOXcRTN3DKXVZWRn0bPwEC4nnMw8Mum3ETrVu3JtBIE4hI4V/+8hcYm2gV1l+8+eab3OGRiTCKLYfNi/Xr17NYsEuYbuAOiFyIkvnz56empqanp//1r39lR0bc6rYLwAMSLWEGwZOcdSZNGCc2TQRY4iP2KxV79+595513qACFBd6lS5cvv/ySs4KYoDGDY+NGHJIz8WG2AuOQpGzYqFEjIrIOJXu6GkghbSGwj8ZzAvWYw/Lwg4mOpDCq8BcRrl3ieQQFecwHQmV/hENm3eTceOLeuOWWW5h3UBJI7GviTbiTeTwSOOzZs+d//Md/PProo4F1eM4z+wgsiY7HrzFREhMEIYH5ReAhAQjxCgkb4P/+7//OPm7r1q3FTIEp3/PPP4+nICKLD2IfBAmEaYnRfvXVV/gsdkOFV2WyABD2XKggKBQ4TxV8Ctu0osLUqVO7detmOzKmKiyaxDRHVIicy9oQx8eTLbJMM86CFNITC1ejeOrwRJkxY4a4WaJTVQqjCn/Byj8UCc/VTp06sWXAs3HevHnsXAq/QE3cB7skPCTZJcWbEMihkArLly+HsYkdRPZQ2I/kKU0XxEdxTKzoAm3HEoYbUsQR7YbRMdzSuAP7YwG4AOSwt0ouiMgTOvfr1++JJ55gFFHj3XffnTt3Lk9vAp8gGjly5B/+8AemP2y10ARfQ00iptz5q1evxnUiYeDAgeSBs61Q4D/84Q8/++yzwsJCNl+ffPLJevXqvfXWWwgUdM011/DEwDLiMHIehQ2BJlxbZMkGnAUppCEQ4kf2hJpd/JYtW8aipBRGFf7CnrEHouLZSzhw+/btPLS5q9m5DAxhcM8we+fuYtIlWuE1CAriXMQhOTcVFWCgiRMnHjt2jI1Ybh4ObSLEyDyffWm7JGqGfRAUtptj5T179oDCLiHYwW3M057NFFHIPiW+AE/B8hJHM2TIEAKiEyZMwJVQgbu6tLQUCPgR4lU4Pk6xw5qamspZm0KBcwqXKqYnSGBKFYiayQWuDW2pViNFYUN2u5LEX4AUqtGG6iuw3U6kjH650phZw8RCUhhV+Isa3wcPQFst27x587vuuivwQVpt1YAT3L24noCCOLJEMSHXO4gOuIshySAbglFqBuu6QZQJBCmkrDuHHRFEe+WVV6hcu3btkpISZtbwsZAURhX+QsqBRUA+duzYKVOmsOkQoU7gKWYoxBTtd6sCT8WDZ0cTiodkWeAsbn/+85+7okmoDcFI7NkV4ZoLASmklZLMrzMyMpiSo1VeXl7YN545JUVSGFX4CyntI1RmnZaTk0MsIEId+xSeguU903W7xLuMFHCiqsSDArdLogZukg2jNoJWDVnYisgUnsJeqqvUUIW/cGU9IoxCeI99VnYfxWGEnKAAGw3sR0So4+4p5nWQuzJtac6BExwR81W7bdRMWBuC0V+PRG3SWBpu3bp1+PDhSGCIWYmIt4o5jIVoy4CSOySX/QWLBSJt7O3dcccdbBYIJdxajwhpubm57FMIPkI+evRookERKrh+inkd5LpYW6BD4Hb92JmwNgSjvx6J3bayEpg29u3bl50RGnIluHhtM6DIdEgu+wuCMUTa2JVgyc2+gFBCLLcEryyPPQ4kqyq7uZBsK53rh7UhGM+cOaOz2m7pBlLILWkxyikuLl6yZAlCWJzix2HcIimMLvuLZs2a4QhBcvjwYfsplCSXF5AhsJtNYDzu/16R2jFm555gnOizqKiIvXzBu5IzoM7luOwvrrrqKrZ5brzxRqZM9kuK4k8wnevk0ZrXWsmjyjtXG5Qp/u8V/dNeCv7nAdyvX79jx47RF4v9Tp06wbhIDKhzaS77i08//fTiiy/evHnznDlzBg4cKPSoqKgoKytbs2aNiJMdPHhQvBxZXl5OvIM6IuLCNGTv3r0cVlZWHj16FIaGYrIkKlAiGAKoGopC+a+//holTQUozM7Q7Nu3z8gRFADtEQTp//zP/yT8En377bc/+ugjtGLCTgQaxq0bh7ty4cKFK2S+FO2yvwAMUwyRV1ZWwiQPEbWGjMcLRsh4mAAEJgSTQDp06NCLL74oFCgoKLjyyisFn6jcZX/BZGnXrl0dOnTo2bNnfn6+QMWEp3Xr1m3btsVBUtKoUSOxAGvevLn4MoWI0LKQES9uN2jQoGHDhtSkoXj1WFSgRDBsuGgoCm2vv/56lDQVoDA7o8OWtpEjKADaIwhSrtvEXqJM0sVUunv37k899ZS7ZgcdN+zt3/8eHfAjkMv+goj6O++8QyB35cqV9t+DiE2gCEqYcYr1CGQGlggowHjkyJEIFaRO6VwZpFACNWR1X1paigJ4rqlTp8ZpsiOF0WV/AbZQcuvFklDJWpUw5YG0UikeyoAx8M/b4tGFJjJBCiVKGTah+vbtK3p/9dVXxdRbHLqbS2FU4S/EjM5dkBpKYz0FaaiYuyqBkZC2uzL1lAZSKFG6DRs2TET3O3bsmJGRET81pDCq8BeEneOHVh/JjC6kjz5x0gSMYkUdJ/n6iAUplBB9vvzyy0mTJtH1pZdeWlRUFKeVCPIhKYwq/AVhJNQynojFQophqu8OjFIzWPUautUjSCG3pDmXc+rUKSYUVVVVNBk1alTQ91AodJekMKrwFwK5uyA1lMZGPaShYu6qBEbIXZl6SgMmpF63sWPHik9PtmnTRsHXW6QwqvAX3333nXqjq+9xt5XU96u4R1BKvUGsWD0XuwMp5KJAJ6LKyspefvllahIkKikpIYePK0lhVOEvkmQ90txKcR1aHYSDspH/e0XxGYlz585lZmaesf6cb/DgwWlpafHp53tSGdDvHUc8UOEvzp49G1GHRJ50se8TVnJRoJ6iQJkkL9SAFFI5CpMnT161ahU9tmrViv0RGAUkhVGFv8BrKoCd8C5YdkEJVyPeCoDx9OnT8e5FB/kghZRpsmPHjry8PNFdcXExOyOCj3cuhVGFvxDvdMcbdsLlM0uHEq5GvBUAo/97Ra4bmT2Bvn37ijcP+vXrl56e7noX1QlkQKs7FVquwl+wPxTasXklB6xkHq4gRKAUf1sdVG7eIUghNbhKS0sXLVpEX82aNWN/BEYZSWGM1V84QXXRRRc5qeb1OnWt5HUUNeoPyiSZMIIUqtEgsVfYt2+f/Ueo06dPV/y6vRRGFf5CwZ5Q7GMWuwS2gaDY5WguAYxJ4i9ACikYjpycHPEnfL169erSpYuCHgO7kMKowl+IVVmgikbyO61kJLRAUKCUmsEGtvUWD1Io3jp/+OGHc+bMoRfiCBMmTIBRTFIYVfgLKQem2FgudtfCSi4K1FMUKP33wd0amqNHjz733HNC2sSJE3EZgq82j8MJBtS5VBX+wrk2fk3fAkllgUGDBlVUVAC5a9euPXv2hNGcVPgLfz2i+UUgpR7TV389ImWx6iovXry4uLiYswQ4CwsL4/pHqPRSHTGg1Z0KLVfhL/z1SKjdvVvC9NVfj8Q+fDxEs7KyhJyCggK2UQWvPmdAnXeqwl8kyfudp6zk3PQerQlK8QcOntRfRmmQQjItJOrm5+dv27aNBunp6dnZ2TCJIimMKvxFklxeh62UqFFX1i8oj/u/VxSbudeuXSt+/Y+daZYkif0AHQPqHI0Kf6HsTXjnsONR81orxUOyVjJB6f9eUSwjwuMzIyNDfHVi+PDhrVq1ikVa7G0ZUOdCVPgLDORcIe/WZG8M8q7+DjUHo9RfNDoUq2E1kEKuKzZu3LgNGzYgNi0tjf0RmMSSFEYV/iKx5lDWO/FtSFl3ienoggvACF2QBAmYkLtAN2/eLH6jjDVISUmJ1Ld23dXEliaFUYW/0MEotnXixzSwUvzkayIZlEmy4QVSyEWzswbJzMwUXwMYOHBgmzZtXBQetSgpjCr8RZJ8XmWvlaIeNq80BKX4YwevKBy1niCFom4e2rCwsFD8WGlqauqIESNCKySkRAqjCn/h/15RQq6DOHXauHHjevXqxUm4VmJBCrmlUnl5+ZAhQ4S0mTNn6jNHk8Kowl+wVBNmMjtn2QXVgNH7p8GYJH9wDFLIlRGrqqrKzs4W+9BZWVkdOnRwRawrQqQwqvAXJ0+edAWY5kJ2WklzJWNXD5QHDhyIXY7+EkAKuaLnrFmz5s+fj6imTZuyPwKjD0lhVOEv9Jl6xXWQWlgprl3oIByUUjNYHXSOTgeQQtG1DWyFe83NzRUl06ZNa9iwoeA1yaUwqvAXTMY0MU1c1SD6DcW1Cx2EgxHSQZN46wBMKPZeBgwYcOjQIeT06NHjgQcegNGKpDCq8BdSHyCuwZQan95tJY0VdEc1UEq9QexOr4mQAlIoxp4/+eQTFiMISUlJmTx5MoxuJIVRhb9IkvVIcyvpdjW4rg8oE/JZF9eB1CgQpFCN1SJUOHbsGGFOUeH1119v0qSJ4LXKpTCq8Bf+7xVpdX3EqMyJEyeS5IUakEKxmGvo0KHi6d25c+fevXvHIip+baUwqvAXSfL37Cy7oPiNqyaSwSjeUNREn/ipAVIoavnLli0juknzunXrzpgxQ+q1a1opo4gYg7VQ4S/q1KkT3K2Jx8zSIRORfQ8TGP3fK/qeRcIdMAXLzMwUZ8aMGdOyZUvBa5gzoM61UuEvpD7I4Vx13WqybQbpppXr+oCRZbnrYjUUCFIoOsVGjRq1ZcsW2rZr165///4w2pIURhX+wv+9Im2vlSgUY3adJBNGkEJRmGj9+vWvvPIKDWvXrl1SUqL59S+FUYW/SJLXh9kGgrhKzCYwJom/ACkkO5pE9zMyMkTMLi8v76abbpKVoLi+FEYV/sJ/H1zxFRDX7nbu3Ck1g42rMnEVDlJItovx48evW7eOVngK9kdgNCcpjCr8hZQD09y4EdRrYaUIFcw4BUr/ffDqhnLr1q35+fmcZTeElYgn/jKbAUVhh+S+v1iyZMndd9991113ffjhhw6V8Kv5FjDAAufPn8/KymJnBCy5ublEOmEMI5f9BXu5zMfmz5//2Wefde/eXRjLX48IO5iRM31VuB5JpM1ACjnXoLi4eOnSpdRn93T06NEwniApjC77i5UrV1522WXdunV76KGHvv32W2Evfz0i7GBGzvTVX4+EDuWePaqxzMwAABAASURBVHvsj/cWFRVJbTqESlNZwoA6785lf4GP+Prrr+fNm8fEzP7imIgVO9fJozVPWcmjyjtXG5RJ8sF3kEJOLFNVVfXcc8+J11L69OnTqVMnJ600qeMQo9DWZX+RkpJy++23s4FK/OKrr74SfVRUVJSVla1Zs0bMYw8ePCheWS8vL2fJRx0xI+IqFJ8SrKysFN84p6EAIyrYNVngaChq9+7dhLtQ0lSAwuyHDx/et2+fkSMoANojCNINGzY4uUTfe++9uXPn0rBJkyYDBgyACRJFiYZXO3flwoULly9fjnoOyWV/0bZt202bNtH3+vXrr7/+ehjIE1Fi9IyRrr32Wi6XGIW43dx9ecCU+qK0+xqokghSJwN66NChnJwcodSUKVN0+xyOUCxCLrW6dNlfXHXVVQ8++GD79u2HDBkyfPhwoSWTjtatW+NKhGaNGjUSq7vmzZuLT3uKFVTt2rWbNm1KEy5HYXQGTLwaJCpwSjAERDQUdfr0aSCgpKkAhdmZ+l100UVGjqAAaI8gSOvXr1/jJfrSSy/t37+fVgT4H374YZ0v0csvvzzwxuGuZOnEbYjyDsllf0GvLOSIEi9evPi6667jMHmILXfIeLxghIyHCUBgQjARaMGCBaWlpVTgITd16tQa61NTN5LS2X1/EWoO8dQNLTeshCsGMgxUKBww8pgKLTevBKRQBFzHjx9/5plnRIVXX31VzI7FoYfyyBiDgKjwF+INlqCOPXIooSbBWkiigTergtH/vSIxdMOGDROR+I4dO2ZkZIhCz+UMqHOdVfiLJIl3sjKEnJveozXB6P9eEWO3cuXKSZMmwVx66aVFRUVSs3pa6UMMqHNlVPgLETFyrpNHa7LsgjyqvHO1wch+ufP63q0JUiis/mzzZ2ZmVlVVcXbUqFGpqakwHqXqMIaFo8JfsBcdtm/DCpmaQoaBCoUDRvHyS+gpw0pACoUFNXbs2I0bN3KqTZs24oULeI9SdRjDwom3v/hHp0kSHmMjDfoHYKP/gVFqButdY4AUCtW/rKzs5Zdfppx5VklJCTm8dyksxurgqPAXYtpWnQbGlJ+3kjFwqgNioTxf3VmTysMiPXfuHCuRM2fOgHTw4MFpaWkwniZgOtdfhb/47rvvnCvk3Zq7reRd/R1qDsrDhw87rOzpaiCFgiBMnjx51apVFLZq1Yr9ERivUyjGCIhU+IskWY80t1IEW5txCpSNGjUyA0tkFCCFAuvs2LEjLy9PlBQXF7MzInh98ig0CcIYWYIKf3H27NnISphx9oSVzMASAQUok+SFGpBCtilYVvft21cE7/v165eenm6f8jQTiLFGICr8BUu+GvUwoALLLsgAIJEhgPH06dOR65hxFqSQjaW0tHTRokUcNmvWjP0RGDMoEGONiFT4C/E3YzWq4vUKzNIhr6OoUX8w1q9fv8ZqBlQAKSSA7Nu374UXXhD89OnTTXpjzcYo0EXOVfiLU6dORVbCjLMHrGQGlggoQCk+DBOhjndPBWoOUkiU5OTkHD16FL5Xr15dunSBMYZsjE4QqfAXF110kRNVvF6nrpW8jqJG/UGZJBNGkEIY5IMPPpgzZw4Mj+IJEybAmEQCo0NEKvyF119ocWhKtoEgh5W9Ww2MSeIvQAoxrSC6KcZr4sSJuAzBG5OD0TkWFf5ChJSd6+TRmjut5FHlnasNSqkZrHPJutUEKTRo0KCKigp069q1a8+ePWEMIzA6R6TCX0g5MOeq61azhZV008p1fUAZ4X1w17tLoECQbtu2rbi4GB0IcBYWFnr3j1CBUB0Bs7pToeUq/EVor36JbwH9LcC8OCsrS+hZUFDANqrgkzlX4S+wezKYmHkdZDxSMCbJemTAgAHbt29nQNPT07Ozs2GMJAbUOS4V/sJfjzgfD/1rMn1NhvXI2rVrS0pKGA6CuyxJDP6GCwMKTIekwl8kyfudp6zk0O7VVdO/HJTirzP1VzVqDQGYkZEh/nBz+PDhrVq1ilqU/g0ZUOdKqvAXWN+5Qt6tedhK3tXfoeagPH78uMPKHq02bty4DRs2oPxPf/pT9kdgDCYG1Dk6Ff7CjD/jq9Gm11qpxmperwDKlJQUr6OIoP/mzZtHjhxJBdYgb775ptTn6mjlOWJAneuswl8kyfziqJWcm96jNUEp9ReN3oLJGiQzM1P8QV2/fv3s3+jzFgopbRlQ5/VV+Avn2sReM4ES2JyHEqiAmq7BCKnpS30vhYWFK1asoN/U1NShQ4cajBSMgqQwqvAXxs/ohN0bWEnwBuegNHXDq7y8fMiQIWLsZs6cyUQdsOLQ4FwKowp/kSSfV9lrJYMvLAENlEb+XlFVVVV2drYI5WZlZXXo0AGkkEBtcC6FUYW/8H+vyKSrrXHjxvXq1TMJkcAya9as+fPnwzdt2pT9ERiQQjBmkxTGYH8RD9MQZ46HWN1ksuyCdNPKdX3AaN4fHB84cCA3N1fYatq0aQ0bNoQHKQRjNklhVOEv/PfBTbrgdu7cyd1lEiKw4CwOHToE06NHjwceeAAGAikEYzZJYVThL0wNjwVdRi2sFFRo3iEopWaw+lvgk08+mT17NnqmpKRMnjwZRhBIIcEbnEthVOEviCQZbG4bGlv3kH1oKgNGyBh0x44dI8wp4Lz++utNmjQRPDkwIZg4kgaipTCq8BdSHyDWwIBRqrDbSlE29k4zUEq9Qaw5MjZQQYSSnTt37t27N4xNlEP2oamMFEYV/iJJ1iPNrWTqVWXjAqUx36RbtmxZYWEh0OrWrTtjxoygN5dACnHWbJLCqMJfnPV/r8igK+7EiRNmvFADiszMTDEyY8aMadmypeDtHKSQfWgqI4VRhb9Ikr9nZ9kFmXpV2bjAKP68wi5JJBND3yNHjtyyZQsC2rVr179/f5ggAikUVGjeoRRGFf6iTp065lk5FBGzdCi03LASMBrwe0Xr168XL2XVrl27pKQk7E9egBQybPhC4UhhVOEvpD7IEYrHKyUHrOQVbaPWE5TsKUTdXIeGLJAzMjLEtDcvL++mm24KqxVIobCnTCqUwqjCX4R13iZZXGAhZgYJ3uAcjF6fMI4fP37dunWMEZ5i6NChMGEJpFDYUyYVSmFU4S/Me3047OXCNhAU9pS+hfKagdHT/mLr1q35+fngZjeElUiEP24CKURNs0kKY1z8xezZs6+++mrbyv774LYpDGB2evl98PPnz2dlZbEzwkDk5uYS6YSpjkAKVXfWmHIpjO77C5aF7733XuCmrpQD8+4wtLCSd/V3qDkovfs+eHFx8dKlS0HK7uno0aNhIhBIoQgVzDglhdF9f8Hk4tFHH2WyZ4Y1fRTGWGDPnj32x3uLioqk1u3GGCFGIC77CzG5+Ld/+7dAtUxejwTgZF4HBRSYyYJRKqKuiRWqqqqee+45sbPTp0+fTp061agYSKEaq3m9ghRGl/3FW2+9FTq5OH78eFlZ2Zo1a8R1dvDgQfFKWXl5OetJzC00PnPmjPjUT2VlpfgGaUVFhdiLFRXsmjggDUUxSxcrL1MBCrMzfSXe6bkRZI08d+5cLqEmTZowy3ByXYGUabJJl6gYQYwgLlHuyoULF1ZWVlLikFz2F5s2bXrzzTfvu+8+otADBgwQSgiLC97gnEsQMhiggAbGs2fPCt4r+eHDh3NycoS2U6ZMSXH2ewgghUQrg3Opt3Vd9hcFBQULFiyYP3/+j3/84wkTJggrN2zYsHXr1m3btuUJTEmjRo3E0pGYqPj0Fo6c8tq1azdt2hSmQYMGNIG59tpreZTBiAo2w2NcQ1FE3cWbJqYCFGbn3uOp660RfPXVV/fv38/1071794cfftjhdQXSK664wqRLVIwgdhCXKHcl67If/OAHlDgkl/2F3evq1attPtLvFdmVvM9wFULex1EDAjA6fD7XIEjVaR5gpaWl9MZzaOrUqTg7eCcEUshJTU/XkcIYL38RaEECE4GHpvLEXCBT0dm4wCiCF3aJzgyxs2eeeUZoyCxDTGDFYY05SKEaq3m9ghRGFf7C6wZ1qD8PLshhZe9WAyPkFf2HDRsmguUdO3bMyMiQUhuYkFQTL1aWwqjCXxCY8KIdZXVmugvJtvJcfTCyDPaE2itXrpw0aRKqsiIuKiqSujFoBVIIxiAKA0UKowp/QSAwjJrGFbEZDBkHKxgQGD3xe0VsbWRmZlZVVQFg1KhRqampMFIEUkiqiRcrS2FU4S8i/EmPF+1bnc5s2UDVnTWmHIye+L2isWPHbty4EbO3adPG3trn0DmBFHJe36M1pTCq8BdiR8qj1nSuNssuyHl9j9YEo/5/cFxWVvbyyy9jYVQtKSkhh5clkEKyrTxXXwqjCn9x8uRJzxkxCoWJq0FRNPRWEzCKl2u1VfvcuXOsRMSu3ODBg9PS0qJTFaTQ99saeCSFUYW/8Ep4LMZroYWVYhSif3NQSs1g1SOaPHnyqlWr6LdVq1bsj8BERyCFomvroVZSGFX4CxFz8pAFo1P1vJWia+uhVhbK89oqvGPHjry8PKFecXExOyOCjyLXHGkUiMI2AWbY8rCFKvyF1AeIw2rpicLdVvKEqrEoCcrDhw/HIiF+bXky9e3bV6x/+/Xrl56eHktfIIVikeCJtlIYVfiLJFmPNLeSJy6RWJQEZaNGjaKVEN92paWlixYtoo9mzZqxPwITC4EUikWCJ9pKYVThLzz354zRDfMJK0XX1kOtQKnnCzX79u174YUXhCWnT58e+6YvSCEh0OBcCqMKf0G82mBz29BYdkH2oakMGKX+AlqZHXJycsSfQvTq1atLly6x9wtSKHY5mkuQwqjCX4i/SdfcarGrxywdil2O5hLAqOHvFX3wwQdz5szBdKhnf0iBw1gIUVAsEjzRVgqjCn9x6tQpTxguRiUPWClGITU1T/x5UIqv2iVelX9qwLSC6KY4mjhxotQNIFqFzUEKhT1lUqEURhX+QnxFxiQTh8VS10phT5lUCErdJoyDBg2qqKjAyF27du3ZsyeMKwRSyBVROguRwqjCX0T3Nq7OJg6rG9tAUNhTJhWCUSt/8dlnnxUXF2NhApyFhYWyf4RKw+oIpFB1Z40pl8Kowl+I/XBj7FsdkJ1Wqu6sMeWglJrBxhU4l1ZWVpbooqCggG1UwbuSgxRyRZTOQqQwqvAXUg4szpaNo/gWVopjB3qIBqU+74Pn5+dv374dw6Snp2dnZ8O4SC2s5KJAPUWB0rliKvyFc238mr4FnFtg7dq148ePpz7rI5YkSfJn0OBNIKnwF0waE4hQWdfM6yBl3SWqIzDqsB45c+ZMRkaG+NuH4cOHt2rVynWDgBRyXaxuAqUwqvAX/npEt0skFn2YvuqwHhk3btyGDRsAkpaWxv4IjOsEUsh1sboJlMIo6y+iAZsk73eeslI0BvJUG1DybE+syps3bx45ciQ6sAYpKSmR+uILrRwSSCGHlb1bTQqjCn+R8MtLzVgetpKavhLYCyiPHz+UWEiXAAAQAElEQVSeQAVYg2RmZop30gcOHNimTZs4KQNSKE7C9RErhVGFv4jlGwT6mLVGTa61Uo3VvF4BlIn9vaLCwsIVK1ZgxtTU1BEjRsDEiUAKxUm4PmKlMKrwF0kyvzhqJX2ugzhpAkqpv2h0V43y8vIhQ4YImTNnzoxraAykkOgrUbmCfqUwqvAXCjDr0EUtK+mgSVx1sFDWimsX1QmvqqrKzs4Wq6GsrKwOHTpUV9OV8gQidUV/h0KA6bAm1VT4iziFo9BeK2pgJa1UiocyoIzrUz2CzrNmzZo/fz4VmjZtyv4ITFwJpFBcu9BBuBRGFf5Cz8+ruD5Ue63kuljdBIIyIb9XdODAgdzcXGGNadOmNWzYUPDxy0EKxU++JpKlMKrwF/7vFWlyZbiiRuPGjevVq+eKKCkhOItDhw7RpEePHg888ACME4qlDkihWCR4oq0URhX+gk1yTxguRiVZdkExCtG/ORjV/8HxvHnzZs+ejXHYmpk8eTKMAgIppKCjxHYhhVGFv/DfB0/sBeFu7zt37mRp4K7MyNKOHTv27LPPijqvv/56kyZNBB/vHKRQvHtJuHwpjCr8RaLCY4pHooWVFHeqvjtQSs1gY9eQDVTxzfvOnTv37t07doEOJYAUcljZu9WkMKrwF2yDedeazjU/byXn9T1RM1RJC6W63ytatmxZYWEhatStW3fGjBlSm3+0ioUUI41F1VjaAtN5cxX+QuoDxM5V160mz0BIN61c1weMUm8Qx6IAO2uZmZlCwpgxY1q2bCl4NTlIITV9JbAXKYwq/EWSrEeaWymBA6+ma1C69UHdGhUeOXLkli1bqNauXbv+/fvDqCSQQip7TEhfUhhV+Av/94oSch3EqdMTJ07w2I+T8ECx69evFy9lEcAvKSlR/9VokEKBKhnJS2FU4S+S5O/ZWXZBQZeUeYdgFH8bGldoPGMyMjLElZOXl3fTTTfFtbuwwkEKhT1lUqEURhX+ok6dOibZtzoszNKh6s4aUw5GBb9XNH78+HXr1mE0PMXQoUNh1BNIIfX9Ku5RCqMKfyH1QQ7FxnKxuwNWclGgnqJAeezYsbjqtnXr1vz8fLpgN4SVSKLeDwYphBpmkxRGl/3F6tWrb7vttvbt2z/++ONMKYWh1a88Rb+Kczb8IMWdqu8OjHGdMLK9l5WVJUIkubm5RDrVYxQ9ghQSvMG5FEaX/QWx1sWLFy9dupStr48++khYOZbXh4UET+RsA0GeUDUWJcEYV39RXFzM9YOGXEKjR4+GSRSBFEpU78r6lcLosr+45pprxMVETNueVvjvgysbewUd7Yzn++B79uyxP95bVFQk9ehzHTtIIdfF6iZQCqPL/kLYAg0WLlzYrVs3cSjlwEQTL+YtrORFzaV0BmWc3gevqqp67rnnRHCkT58+nTp1klLM9coghVwXq5tAKYzu+wvGu3fv3qWlpfb8Yv/+/WVlZWvWrBGRlYMHD4ot3/LycharmA//Qn7mzBnxp/iVlZXiG2EVFRUiVioqUEcwTFh8UQm0FUMcjxH805/+NHfuXEa5SZMmBQUFCQSYJJcodyXPdfGzDJj9H1TTP5f9BTHOxx57jOD2DTfcYHctYlf2oakM7m/Xrl2morNx4bLFdyjsEleYI0eOPP/880LUlClTrrzySsEnMAep1LvSCVQ1lq6FX3YowWV/MXv2bLZIRo0a1bFjx3fffVcoQRC0devWbdu2FfNY9nvFupRy8WkMMSMi5NG0aVOaNGjQQHw96dprrxXREFGBU4JhgaOhqBtvvPGWW25BSVMBCrMzBNddd53rIzh+/HgxZ+zevfvDDz+sw8UAUjb7TLpExQjalyh3JYu+e++9lxKH5LK/eOKJJ5jLsUUCPfroo0IJ8Zae4A3OWTpBBgMU0MDIylHwbuULFix48803kcajYurUqbVqJeZ7wigQSCCFAkuM5KUwuuwvwhrU9csrbC8JLzxspYSrEW8FQCm+0O1WR0h75plnhLRXX31VzDHFYWJzkEJyOniwthRGFf7C/70iD15F1arMIjElJaXa0/In8vLyiBTQjjVsRkYGjCYEUkgTZeKnhhRGFf4iSeYX7OlA8RtXTSSDUWyOuKLPypUrxfc4eagUFRXVqqXFSkRAAykkeINzKYwq/IXBtg6EVstKgSVG8hZKd+5qVs6ZmZlVVVUYihh5amoqjD7kIlJ9QIVqAszQwupKVPgLYt3VdW9SObE6yCREYbGAkTB72FOyhS+//PLGjRtp1aZNmwEDBsBcoNM/kEI6aRQXXaQwqvAXSfL+BfvYUFyGVCehYDxy5EjsGpWVlY0dOxY5F198cUlJCTm8VgRSSCuV4qGMFEYV/iJRf48cD+NGkNnYShEqmHEKlLH/XhFb7IQ2RWBr8ODBaWlpGhoHpJCGirmrkhRGFf5CvPHiLkgNpbHsgjRUzF2VwBj7XGDSpEmrV69GsVatWg0bNgxGQwIppKFi7qokhVGFvzh58qS7CPWUxqYglGDd4t89GMWLmFF3tWPHDttHFBcXszMStai4NgQpFNcudBAuhVGFv3ArPKaDcSPo0MJKESqYcQqUUjPYINTshvTt21c8Qvr165eenh5UQZ9DkEL66BMnTaQwqvAXXCJxgqqV2PNW0kqleChjoYz+94reeOONRYsWoVizZs1EvBNeT4oRqZ6gQrUCZmhhdSUq/IXUB4irU1T/8t1W0l/PGDUEpdQbxIHd7du378UXXxQl06dPjz1uKkTFKQcpFCfh+oiVwqjCX3hpPRLDMDa3UgwCvNEUlI0aNYpO15ycHPE2Ya9evbp06RKdEGWtQAop6y5RHUlhVOEvzp49myhbqOz3hJVU9piQvkAZ3Qs1H3zwwZw5c9AZdzNhwgQYzQmkkOZKxq6eFEYV/oLN9thR6S+BZRekv54xagjGKH6viGkF0U3R9cSJE3EZgtc5Bymks4au6CaFUYW/EN+8cQWbzkK4ByCdNXRFNzBG8XtFgwYNqqioQIGuXbv27NkTRn8CKaS/njFqKIXRbX8RTvdTp06FKzat7ICVTEMVggeUxyR/r+izzz4rLi5GEgHOwsJCqT9wolWiCKRQonpX1q8URhX+wv7wrzITJKSjulZKSNcqOwWl1ITx5MmTWVlZQsOCggK2UQWvfw5SSH89Y9RQCqMKfxH768MxWkRNc7aBIDV9JbAXMEr5i/z8/O3bt6Nwenp6dnY2jFcIpJBXtI1aTymMKvwFT5iowXio4U4reUjh6FQFpfMZ7Jo1a8aPH09HuBiWJN76SyKQQiivMbmgmhRGFf5CyoG5YIAEiWhhpQR1rq5bUDp8H/zMmTOZmZni9cHhw4e3atVKnZZu9ARSyA1JWsuQwqjCX2htLV+5uFlg3Lhx4rdw0tLS2B+JWz++YHUWUOEv/PWIuvGMf09MX52sRzZt2jRy5EjUYQ1SUlIi9UfTtNKBQArpoElcdZDCqMJf+OuRuI63YuFMX2tcj7AGYU9EvNY1cODANm3aKFbScXeRKoIUilTDiHNSGFX4iyR5v/OUlYy4hCKBACWBiUg1LrigsLBwxYoV1ElNTR0xYgSMFwmkkBc1l9JZCqMKf1Hj5SUFT9vKh62krXpuKQbK48ePR5BWXl4+ZMgQUWHmzJnenV2CFBJADM6lMKrwF9p+QMndi+BaK7krU0NpoIzwe0VVVVXZ2dnCobAk6dChg4YQHKoEUshhZe9Wk8Kowl8kyfziqJW8e9041ByUAX/RGNxo1qxZ8+fPp7Rp06bsj8B4l0AKeVd/h5pLYVThLxzq7fVqtazkdRQ16m+hDP97RQcOHMjNzRUSpk2b1rBhQ8F7NI+A1KOIwqoNzLDlYQtV+Asv7qWFNVbkwgZWilzHgLOgrC4kgbM4dOgQGHv06PHAAw/AeJpACnkaghPlpTCq8BfRfV7FCVSt6uy1klYqxUMZUIb9vaJ58+bNnj2bHoluiJ9Ehfc0gRTyNAQnykthVOEv/N8rqn7YvHemcePG9erVC9L72LFjzz77rCh8/fXXmzRpInhP5yCFPA3BifJSGFX4iwsvVNGLE9PEtQ7LLiiuXeggHIyhf3DMBqr4bGznzp179+6tg56x6wBSKHY5mkuQwqjiTvbfB9f8ipFSb+fOncQ1A5ssW7assLCQkrp1686YMUMqfkYrbQmkkLbquaWYFEYV/qK68JhbgDWR08JKmigTPzVAGTiDJTiVmZkpuhszZkzLli0Fb0AOUsgAIJEhSGFU4S+qqqoiaxzDWY2anreSRgrFRxUL5Xlb9siRI7ds2cJhu3bt+vfvD2MMBSE1BlcQEGAGlUQ4VOEvpD5AHEFXzU+xgIc0VzJ29cBov0G8fv168VIWa+CSkhLDPrwIUih2i2kuQQqjCn+RJOuR5lbS/OKIXT1Qii9Knz17NiMjQ/wxYV5e3k033RS7cK0kgBTSSqV4KCOFUYW/4MKKB07dZJ6wkm5aua4PKIlZIHb8+PHr1q2DwVMMHToUxjACKWQYqFA4UhhV+AvxCApV1LASll2QYaBC4YDx9OnTW7duzc/P5yy7IaxEjHzFBqQQGM0mKYzu+4vBgwffeeedbMLb04o6deqYbXGBjlk6JHiDczDWq1cvKytLzDJyc3OJdBqJF6SQkdACQUlhdNlf/PWvf927d+/nn39+4403ih/LRDOpD3JQ36N0wEoeVd652qDcuHHj0qVLacLu6ejRo2GMJJBCRkILBCWF0WV/sXLlynvuuQdt7r33XvGFpbKysiR5X+sbK4HdbFqzZo0IWwCzqKiobt26MEaSNZ7fuAtNQ2nAPOb4B+tc9hdHjhxhsopR6tevb++67d+/vzgJ0p+tZDzQkdZXfBniPn36dOrUCcZUusxKpqKzcYHS5mtkXPYXDRs2FL6qsrLyyiuvFN1v2bKF5a7xNMJKxsNctWoVg3v11Ve/9tpr5eXl4m0f8U7xmTNnWI0y6FQQX2GpqKgQq1FRgVOCYcoppsEHDx4U8XkNRYFl/fr1BgNk7r9w4cLly5czLg7JZX9x2223LVq0iL4XLFiQnp4O07p16x/96EftrdTNT0ZY4KqrriLGyfyif//+3bt3//Wvf52Tk0P+8MMPZ2dnwzzxxBMEvGGeeeaZHj16wIgKNvPYY49lZGRw+PTTT/fs2RNGQ1FTpkx59913DQaIzceMGfPOO+8I383dWiO57C/S0tKaNGnC/simTZu4ekT327ZtW2KluX7yLRCbBfzWLlrAuin/kX311VfiVq0xd9lf0N+4cePYH/njH/8Y+lfPnPXJt4BvAe9awH1/EWgLYhm33norEVDbgbFYuv322++44w7WToE1zeAJNTdu3Lhjx4533XUXK3MzQAWhCH2/JqiCGYdmD6V9Y7I1zng5vyvj6y8uv/xyNg0eeeQRdBI0bNgwSmbNmsVlJ0oMyzt06LB48eLPPvtM6jUYrxgh7Ps1XlFeVk+DhzLoxnR+V8bXX7AkCbxtZsYTeAAAAeVJREFUvvvuu4suuqhBgwbNmjWzd1tlR1Hz+itWrCB8k5eXp6eeMWoV+n5NjAJ1bm7wUAbemFJ3ZXz9RdDVcOTIkfr164tCNLZfGBclBuRNmzYluEv4Zv/+/e+//74BiIIgMIKsLilkHE31+KCDjB9KMApiTBlNwdd4V7rpL7799luW7oHEbSP0qFXrH79YYb+dQSHOAuVgvEuheDG9ePvloYceYuruXWjVaW6PYGXA+zXVVfZ0+SWXXGL2UNqjk5KSQjhDHNZ4V7rpL9hJXfz9dPXVVws9xCe2WDWhEJfarl277Le5RAUv5qF47ZejmWLccMMNXgQVWefQ92si1/fuWfGTjuhv6lACDeLGxC06vyvd9Bd0H0pdunT59NNPs7Ky3nzzTc6OHj2akscff7ygoIBDz1FkhYkz/+IXvyB+UVFRAcbIlb14Nuz7NV4EUqPOxg8lt6G4MUtLS53flXH3F+yG7Nmz54svvuhtfWaenVTCSMuWLbv55ptrHDPPVbj33nvXrl3LE+mNN94w9VcUkuT9GuOH0r4xn3zySed3Zdz9hefueV9h3wK+BaqzgO8vqrOMX+5bwLdAsAVU+4vg/v1j3wK+BbxjAd9feGesfE19CyTaAr6/SPQI+P37FvCOBf4/AAAA///gey5dAAAABklEQVQDAG5mhM4C8u40AAAAAElFTkSuQmCC)

# Relu activation function이 속도를 개선, 정확도도 개선

import numpy as np
def tanh_function(x):
  exp_x = np.exp(x)
  exp_neg_x = np.exp(-x)
  numerator = exp_x - exp_neg_x
  denominator = exp_x + exp_neg_x
  return numerator / denominator
z = np.array([-1.0, 0.0, 2.0])
result = tanh_function(z)
print(f"입력 z: {z}")
print(f"tanh(z): {result}")

print(f"Numpy 내장 tanh: {np.tanh(z)}")

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label='tanh')
plt.plot(line, np.maximum(line, 0), linestyle='--', label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
print(np.tanh(-100))
print(np.tanh(-10000000000))
# activation function
# 모든 내적연산 뒤에 activation(sigmoid(0~1), tanh(-1~1), relu(0~00))이 붙음
# relu(0~00) 파생함수들이 출현
# 최종 출력 전에 activation(identity(회귀), sigmoid(이진분류), softmax(다중분류))

def plot_iris(X, y, model, title, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5):
      XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000),
                          np.arange(ymin, ymax, (ymax-ymin)/1000))
      ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T),
                      XX.shape)
      plt.contourf(XX, YY, ZZ,  alpha=0.5)
      plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='o', label='0',
                  s=100)
      plt.scatter(X[y == 1, 0], X[y == 1, 1], c='g', marker='s', label='1',
                  s=100)
      plt.xlim(xmin, xmax)
      plt.ylim(ymin, ymax)
      plt.xlabel("특성 0")
      plt.ylabel("특성 1")
      plt.title(title)
      plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
mlp = MLPClassifier(max_iter=3000, hidden_layer_sizes=(100,), random_state=0).fit(X_train, y_train)

# 100x2 (입력 데이터) 2x100 100x1 (가중치 2개)
plot_iris(X_train, y_train, mlp, "ANN")

MLPClassifier

X.shape # (100, 2)

# 가중치
# 데이터 입력차수 * hidden_layer_size
np.array(mlp.coefs_[0]).shape # (2, 100)

# 2진분류일 때는 2개
# logit은 한 개 -> 확률로 변환 (2개)
np.array(mlp.coefs_[1]).shape # (100, 1)

mlp.coefs_

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[30, 10])
mlp.fit(X_train, y_train)
plot_iris(X_train, y_train, mlp, "ANN")
# 2x30 30x10 10x1 가중치가 3개가 형성

X_train.shape # 입력데이터 (75, 2)

mlp.coefs_[0].shape

mlp.coefs_[1].shape

mlp.coefs_[2].shape

np.unique(y_train) # [0, 1]

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
# 일반함수 - 결정경계를 자동으로 시각화
DecisionBoundaryDisplay.from_estimator(
    mlp, X_train, response_method="predict", # 분류기전달
    cmap=plt.cm.RdBu, alpha=0.6, ax=ax[0]
)
scatter = ax[0].scatter(X_train[:, 0], X_train[:, 1],
                        c=y_train, cmap=plt.cm.RdBu, edgecolors="k")
ax[0].set_title("MLP Decision Boundary (Train Set)")
ax[0].set_xlabel("Feautre 0")
ax[0].set_ylabel("Feature 1")

param_range = np.logspace(-5, 2, 8)
ValidationCurveDisplay.from_estimator( # 파라미터 검증 후 시각화
    estimator=mlp,
    X=X_train,
    y=y_train,
    param_name="alpha",
    param_range=param_range,
    score_name="Accuracy",
    cv=5, # cross validation
    ax=ax[1]
)
ax[1].set_xscale("log")
ax[1].set_title("(Hyperparameter: alpha)")
ax[1].grid(True)

train_sizes = np.linspace(0.1, 1.0, 5)
# 훈련데이터의 사이즈를 점점 늘려가면서
LearningCurveDisplay.from_estimator( # 학습 결과
    estimator=mlp,
    X=X_train,
    y=y_train,
    train_sizes=train_sizes,
    score_name="Accuracy",
    cv=5,
    ax=ax[2]
)
ax[2].set_title("Learning Curve (Data Scalability)")
ax[2].grid(True)
plt.tight_layout()
plt.show()
# dicision boundary : 과적합 상태 (조절 : hidden_layer_size, alpha, solver 교체)
# validationCurveDisplay : 최적의 alpha 값은 0.1
# learningCurveDisplay : train은 항상 1, 완만해질 때까지 개선의 여지가 있음
# 학습데이터를 추가 확보

def plot_multi(X, y, model, ax, title, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5):
    XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000),
                         np.arange(ymin, ymax, (ymax-ymin)/1000))
    ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T),
                    XX.shape)
    ax.contourf(XX, YY, ZZ,  alpha=0.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='o', label='0',
                s=100)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='g', marker='s', label='1',
                s=100)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
    ax.set_title(title)

fig, axes = plt.subplots(3, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 50, 100]):
  for ax, alpha in zip(axx, [0.1, 0.01, 0.001, 0.0001]): # 규제는 약해지면 과적합
    mlp = MLPClassifier(solver='lbfgs', random_state=0, activation='tanh', max_iter=1000,
                        hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
                        # [10, 10] [50, 50] [100, 100] 자세하게 찾음 -> 과적합
    mlp.fit(X_train, y_train)
    plot_multi(X_train, y_train, mlp, ax, str(n_hidden_nodes) + " " + str(alpha))
plt.show()

# 문지방 경계값
# bias variation도 trade off 관계
# precision : recall trade off 관계

from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

# 최적 모델 구성 구조
# MLP모델 -> Calibration(확률교정) -> [임계값 최적화]

import pandas as pd
wine = pd.read_csv('/content/drive/MyDrive/[미래융합교육원] AI빅데이터_파이썬/dataset/wine_data.csv',
                   names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash",
                          "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols",
                          "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"],
                   encoding="utf-8")
# 와인데이터의 경작자 예측
X = wine.drop('Cultivator', axis=1)
y = wine['Cultivator']

y_binary = (y == 1).astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1000, random_state=42)

mlp.fit(X_train, y_train) # 학습하고 문지방 튜닝할 것

from sklearn.calibration import CalibratedClassifierCV
# sigmoid 적용 -> 확률값이 실제값을 반영한 확률값으로 수정
calibrated_mlp = CalibratedClassifierCV(
    estimator=mlp,
    method='sigmoid',
    cv=5
)
# calibrated_mlp.fit(X_val, y_val)

# threshold 기준값 : 0.5 - 2진분류
# 임계값을 높이면 precision은 점점 정확해짐, recall은 평가가 안좋아짐
# 임계값을 낮추면 recall은 정확해지는데 precision은 안좋아짐
tuned_mlp = TunedThresholdClassifierCV(
    estimator=calibrated_mlp,
    scoring='f1', # f1평가 precision, recall의 조화평균
    cv=5,
    n_jobs=-1 # 멀티코어
)
print("최적의 분류 임계값 탐색 및 MLP 학습 진행 중...")
tuned_mlp.fit(X_val, y_val)
# tuned_mlp.fit(X_train, y_train)

# 분류문제 - 확률값으로 결정
# 신경망의 확률이 의미하는 것은 (분류)
# logit 출력 -> logit값 간의 비율을 exp값으로 확장한 다음
# 확률값으로 변경 -> 진짜 확률값하고 맞는가
# 

print(f"찾은 최적의 임계값(Best Threshold): {tuned_mlp.best_threshold_:.4f}")
y_pred = tuned_mlp.predict(X_test)
print("\n[Classification Report]")
print(classification_report(y_test, y_pred))
print("[오차 행렬 - Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))