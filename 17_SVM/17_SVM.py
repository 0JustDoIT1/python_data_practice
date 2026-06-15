import matplotlib.pyplot as plt # 시각화 script
# !apt-get update -qq # ubuntu 설치 명령어 (물어보지 말고 설치해라)
# !apt-get install fonts-nanum* -qq # 나눔 폰트 / 시스템에서 관리
import matplotlib.font_manager as fm # 글꼴 관리
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  name='NanumGothic')
# truetype font (t/f)
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'})
plt.rcParams["axes.unicode_minus"] = False # 마이너스 기호 깨짐 보완

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])
svm = SVC(kernel = "rbf")
svm.fit(X,y)
xx, yy = np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# SVM 이진분류
# kernel function 고차원으로 확대
# 초평면을 구하고
# 분류
# SVM이 구한 초평면식을 이용해서 초평면으로 부터의 거리 decision_function

plt.contourf(xx,yy,Z, levels =np.linspace(Z.min(), Z.max(), 50), cmap="coolwarm", alpha=0.6)
plt.scatter(X[:,0], X[:,1], c=y, cmap = "coolwarm", edgecolors="k", s=100)
plt.title("SVM with RBF Kernel (XOR 문제)")
plt.show()

# 2진 perceptron
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 30

x1 = np.random.uniform(-1, 1, N) # 균등분포
x2 = np.random.uniform(-1, 1, N)
X = np.column_stack((x1, x2)) # 열로 합침
Y = np.where(x2 > 0.5 + x1, 1, -1) # 종속변수 결정

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=np.where(Y > 0, 'black', 'red'), marker='o',
            s=np.where(Y > 0, 100, 100))
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.plot(np.linspace(-1, 1, 100), 0.5 + np.linspace(-1, 1, 100), 'b--')
plt.title("Initial Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()

def calculate_distance(x, w, b): # 데이터, 가중치, bias(편향)
    return np.dot(x, w) + b

def linear_classifier(X, w, b):
    distances = np.apply_along_axis(calculate_distance, 1, X, w, b)
    return np.where(distances < 0, -1, 1)

# 가중치는 반드시 초기화(결과 값에 영향을 줌) - MLP를 이해하기 위해
# 초기 신경망에서는 random값으로 대체됨

initial_w = np.array([-1, 1]) / np.sqrt(2) # 기울기
initial_b = -np.sqrt(2) / 4 # 절편
initial_predictions = linear_classifier(X, initial_w, initial_b)
print(f"Initial Classifier Predictions: {initial_predictions}")

# 정답이 아님 - 가중치가 임의로 초기화 된것이기 때문
def second_norm(x):
    return np.linalg.norm(x)

# 계층이 한개인 신경망 (한계:xor회로를 해결할 수 없음)
# 신경망 -> multi layer (1개의 레이어만 추가되도)
# * layer을 행렬의 횟수로 생각해 볼 수 있음
# SVM -> 고차원 (kernel function)
# 신경망 (뉴톤법, quansi-newton: 기울기-곡률(2차미분))
# 기울기 ---
# perceptron : 가중치
def perceptron(X, Y, learning_rate=1): # 학습률(경사하강법을 보완하기 위해)
    w = np.zeros(X.shape[1]) # 변수 개수만큼 가중치 사이즈
    b = 0
    k = 0
    R = np.max(np.apply_along_axis(second_norm, 1, X))

    incorrect = True

    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=np.where(Y > 0, 'black', 'red'), marker='o', s=5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Perceptron")
    plt.xlabel("X1")
    plt.ylabel("X2")

    print("계속학습을 위해 Enter")

    while incorrect: # True이면
        incorrect = False # 종료 조건
        yc = linear_classifier(X, w, b)
        for i in range(X.shape[0]): # 데이터 행수
            # Y[i] : 정답  != yc[i] : 예측값
            if Y[i] != yc[i]: # 데이터가 맞게 예측하지 않으면
                # 오답 : 가중치 학습중
                w = w + learning_rate * Y[i] * X[i, :]
                b = b + learning_rate * Y[i] * (R ** 2)
                k += 1 # 횟수 증가
                if k % 5 == 0: # 5회전 마다
                    if w[1] != 0: # 기울기하고 절편으로 변경해서 출력(사람이 이해하도록)
                        intercept = -b / w[1] # 절편
                        slope = -w[0] / w[1]  # 기울기
                        x_plot = np.linspace(-1, 1, 100)
                        y_plot = slope * x_plot + intercept
                        plt.plot(x_plot, y_plot, 'r-') # 회귀선을 출력
                        print(f"반복 # {k}")
                        input("계속하기 위해 [enter] 키를 누르세요...")
                incorrect = True
    plt.show()
    s = second_norm(w)
    return {'w': w / s, 'b': b / s, 'updates': k}

# 신경망 호출
p = perceptron(X, Y)
print(f"\nPerceptron학습결과 :")
print(f"  가중치 (w): {p['w']}")
print(f"  Bias (b): {p['b']}")
print(f"  업데이트 횟수 (k): {p['updates']}")

# svm의 기본원리
import numpy as np # ndarray
import pandas as pd # Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer # 유방암(악성, 양)
from sklearn.preprocessing import StandardScaler # 정규화 클래스(pipeline)
from sklearn.svm import SVC # support vector classifier
data = load_breast_cancer() # bunch
X = data.data # 독립변수
y = data.target # 종속변수
print(data.target_names)
print(data.feature_names[:10])

# 참고
# 0	-> malignant : 악성암
# 1	-> benign : 양성암
# radius : 세포핵 반지름
# texture : 세포 표면의 질감
# perimeter : 세포 둘레
# area 세포 면적
# smoothness : 표면이 얼마나 매끄러운가
# compactness : 세포가 얼마나 조밀한가
# concavity : 세포 경계가 얼마나 오목한가
# symmetry : 대칭성

# 2개의 변수 선택
feature_names = ["mean radius", "mean texture"]
idx1 = list(data.feature_names).index("mean radius")
idx2 = list(data.feature_names).index("mean texture")
X_2d = X[:, [idx1, idx2]]
y_svm = np.where(y==0, -1, 1) # y 매핑
df = pd.DataFrame(X_2d, columns = feature_names)
df["target"] = y_svm
df.head()
# SVC 기준: 거리값 기준: 정규화

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_2d)
model = SVC(
    kernel = "linear", # perceptron방식 - 이동최적화
    C = 1.0 # 규제값 : 작게주면 부드러워짐, 크게주면 변동
)
model.fit(X_scaled, y_svm)

w = model.coef_[0] # 가중치
b = model.intercept_[0] # 바이어스
print("w:", w)
print("b:", b)

# svm이 구하는 결과값은 초평면식
# coef_은 기울기
# dicision function : 방정식에 값을 대입
f_values = X_scaled @ w + b
result = pd.DataFrame({
    "x1_mean_radius" : X_scaled[:, 0], # 변수값
    "x2_mean_texture" : X_scaled[:, 1],
    "y": y_svm, # 실제값
    "f(x)": f_values, # dicition_function값 : 초평면으로부터의 거리값
    "prediction": np.sign(f_values) # 예측값
})
result.head(10)
# 초평면으로부터 1 ~ -1 내에 있으면 : 경계선에 있음
# 0 이면 : 초평면상에 있음

# 법선의 크기
w_norm = np.linalg.norm(w)
margin = 1 / w_norm
total_margin_width = 2 / w_norm

print("||W||", w_norm)
print("한쪽 마진 폭 1/||w||:", margin)
print("전체 마진 폭 2/||w||:", total_margin_width)

support_indices = model.support_
support_vectors = model.support_vectors_
print("support vecotr 개수:", len(support_indices)) # 157
print("support vector index:", support_indices[:20])
print("support vectors:\n", support_vectors[:5])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
# 분할하기 좋은 형태로 데이터를 생성
X,y = make_classification(n_samples = 200, n_features=2, n_informative=2,
                          n_redundant=0, n_classes=2, random_state=42)
y[y == 0] = -1 # -1과 1로 종속변수 (SVM은 -1과 1로 분류할 때 양호하게 분류)
X_train, X_calib, y_train, y_calib = train_test_split(
    X, y, test_size =0.5, random_state =42
)

svm_model = SVC(kernel = "linear", C=1.0, probability = False, random_state=42)
svm_model.fit(X_train, y_train)
# 테스트의 초평면으로부터의 거리값
svm_scores = svm_model.decision_function(X_calib)
print(f"SVM 거리 점수 예시 (일부) : {svm_scores[:5]}")
print(f"실제 레이블 예시: {y_calib[:5]}")

# 거리값 -> 확률값 으로 변경해보기
# Platt Scaling
# A : 변환 기울기, B : 변환 절편
# 수식 : P(y=1|x) = 1 / (1 + exp(A * f(x) + B)) # sigmoid

# 거리값 -> 확률값
# SVM : -1, 1 라벨링을 해야 결과가 좋고
# Logistic regression은 확률 0~1 값
y_calib_platt = np.where(y_calib ==1, 1, 0) # 0~1
platt_scaler = LogisticRegression(solver = "lbfgs", C=1.0, random_state=42)

# vm_scores 거리값 y_calib_platt 실제값(0,1)
platt_scaler.fit(svm_scores.reshape(-1, 1), y_calib_platt)
A = platt_scaler.coef_[0][0] # 기울기
B = platt_scaler.intercept_[0] # 절편
print(f"\nPlatt Scaling 매개변수: A = {A:.4f}, B = {B:.4f}")

def platt_predict_proba(scores, A, B): # 거리값 -> 확률
  return 1 / (1 + np.exp(A * scores + B))

calibrated_probabilities = platt_predict_proba(svm_scores, A, B)
print(f"\n변환된 사후 확률 예시(일부): {calibrated_probabilities[:5]}")
# 거리값 그대로는 결정근거로 사용이 불가능
# 거리값 - 라벨링 calibration 조정

# 시각화
plt.figure(figsize=(10,6))
score_range = np.linspace(svm_scores.min() - 0.5, svm_scores.max())
proba_curve = platt_predict_proba(score_range, A, B)
plt.plot(score_range, proba_curve, color= "red", label=f"Platt Scaling Curve (A={A:.2f}, B={B:.2f})")
plt.scatter(svm_scores, calibrated_probabilities,
            c = calibrated_probabilities,
            cmap = "RdYlGn", edgecolor = "k", alpha =0.6,
            label = "Calibrated Points")
plt.axvline(0, color = "gray", linestyle="--", label = "SVM 결정 경계 (Score=0)")
plt.axhline(0.5, color="green", linestyle=":", label = "확률 P =0.5")
plt.title("Platt Scaling")
plt.xlabel("SVM(거리)")
plt.ylabel("Estimated Probability")
plt.legend()
plt.grid(True)
plt.show()
# 기준선 0이 초평면
# 확률 0.5 == sgmoid중심이어야 하나 현재 그래프는 같지 않음
# 이유: 매핑비율 때문에 거리값 / 종속변수 조정

# SVC loss function : hinge loss
# SVR : Epsilon-Insensitive loss

fx = np.linspace(-2, 2, 100)
hinge_loss = np.maximum(0, 1-fx)
plt.figure(figsize=(8,5))
plt.plot(fx, hinge_loss, label = "Hinge Loss", color = "red")
plt.axvline(x=1, color = "gray", linestyle="--", label = "Margin Boundary(+1)")
plt.axvline(x=0, color = "black", linestyle = "--", label = "Decision Boundary(0)")
plt.xlabel("f(x) (결정 함수 값)")
plt.ylabel("Hinge Loss")
plt.title("Hinge Loss 그래프")
plt.legend()
plt.grid()
plt.show()
# 경계선(0)에서 벌점(loss)을 줌 -> 1(기준)까지 내려와야 벌점이 없음

# SVC는: 다중분류 안됨 / but 지원은 함 : ovo, ovr 방식으로 지원은 함

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state =42)

# 다중분류일 때는 내부적으로 ovo, ovr을 선택 적용 -> 명시적으로는 안보이는 형태
svm_model = SVC(kernel = "linear", C=1.0)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM 다중 분류 정확도: {accuracy:.4f}")

# 명시적으로 ovr 사용
from sklearn.svm import LinearSVC
# meta-estimator : OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
ovr_model = OneVsRestClassifier(LinearSVC())
ovr_model.fit(X_train, y_train)
y_pred_ovr = ovr_model.predict(X_test)
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
print(f"SVM 다중 분류 정확도 (OvR): {accuracy_ovr:.4f}")

# C & gamma 제어
# C : 규제
# gamma: 초평면의 모양을 제어

from sklearn.svm import NuSVC
from sklearn.datasets import make_moons
X,y = make_moons(n_samples = 200, noise =0.2, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gamma_values = [0.1, 1, 10]
nu_values = [0.1, 0.5, 0.9]
fig, axes = plt.subplots (len(gamma_values), len(nu_values), figsize=(12,9))

fig, axes = plt.subplots (len(gamma_values), len(nu_values), figsize=(12,9))
for i, gamma in enumerate(gamma_values):
  for j, nu in enumerate(nu_values):
    model = NuSVC(kernel = "rbf", gamma = gamma, nu = nu)
    model.fit(X_train, y_train)
    ax = axes[i,j]
    ax.set_title(f"gamma = {gamma}, nu = {nu}")
    xx, yy =np.meshgrid(np.linspace(-1.5, 2.5, 100),
                        np.linspace(-1, 1.5, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap = plt.cm.coolwarm)
    ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train,
               edgecolor = "k", cmap = plt.cm.coolwarm)
    ax.set_xticks(())
    ax.set_yticks(())
plt.tight_layout()
plt.show()
# gamma : 초평면 모양
# nu : 이상치 허용 비율의 상한 , 서포트 벡터 비율의 하한
# if 이상치(잘못 분류된 것) 허용 비율을 키우면 -> 모델이 느신 -> 일반화
# if 이상치 허용 비율을 줄이면 -> hard margin -> 정확하게 하려함 -> 과대적합
# if 서포트벡터가 많아지면 -> 과대적합
# if 서포트벡터가 작아지면 -> 과소적합

# nu 작으면 -> 과대적합  ------------ 크면 -> 일반화
# nu가 출현한 이유 : 직관적으로 이상치, 서포트 벡터로 제어
# (gamma. C가 직관적으로 제어하지 못함으)
# gamma 커지면서 -> 과대적합 --------

# custom kernel 적용
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import laplacian_kernel
X, y = make_moons(n_samples =300, noise = 0.1, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# laplacian_kernel 절대값으로 계산된
X_train_laplacian = laplacian_kernel(X_train, X_train, gamma = 0.5)
X_test_laplacian = laplacian_kernel(X_test, X_train, gamma = 0.5)

# linear, rbf 방사형 커널 (차의 제곱)
# polynomial, sigmoid
# 미리 계산된 : 이미 고차원으로 계산이 되었음

def custom_poly_kernel(X1, X2, gamma=1.0, c=1.0, d=3):
  return (gamma * np.dot(X1, X2.T) + c) ** d

X_train_poly = custom_poly_kernel(X_train, X_train)
X_test_poly = custom_poly_kernel(X_test, X_train)
# SVC 커널을 사용할 때는 자동으로 test 데이터와 트레인데이터를 묶어서 처리

svm_laplacian = SVC(kernel = "precomputed")
svm_laplacian.fit(X_train_laplacian, y_train)
accuracy = svm_laplacian.score(X_test_laplacian, y_test)
print(f" Accuracy: {accuracy:.4f}")
# 사용자 커널 : cosine, 방정식을 새로 만들어서 제공, 카이제곱


# - OneClassSVM 이상탐지 모델 : 정상데이터로 학습
# - LOF(local outlier factor) : knn으로 구성된 모델
#                               이웃데이터를 보고 희소성을 봄
# - isolate Tree 모델  : 이상한 데이터는 초기에 분리됨 - 초기 leaf

from sklearn import svm
X = 0.3 * np.random.randn(100,2)
X_train = np.r_[X + 2, X - 2] # 두개의 봉우리
X_outliers = np.random.uniform(low = -4, high = 4, size=(20,2))

# nu = 0.01 : 이상치를 1%까지 허용
# gamma = 0.005(작게 줌) : 넓게
# 너무 타이트하게 학습하면 과적합 됨
clf = svm.OneClassSVM(nu = 0.01, kernel = "rbf", gamma = 0.005)
clf.fit(X_train)

y_pred_train = clf.predict(X_train) # 정상데이터
y_pred_outliers = clf.predict(X_outliers) # 이상데이터
plt.scatter(X_train[:, 0], X_train[:, 1], c ="blue", label = "Normal Data")
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="red", marker="x", label ="Anomalies")
plt.legend()
plt.title("One-Class SVM Anomaly Detection")
plt.show()

n_error_train = y_pred_train[y_pred_train == -1].size
print("훈련데이터 오분류", n_error_train)
n_error_outliers = y_pred_outliers[y_pred_outliers == -1].size
print("이상데이터 오분류", n_error_outliers)
# 훈련데이터가 틀렸다고 나오는 이유 : nu =0.01을 허용해줬기 때문

param_grid = {
    "nu": [0.01, 0.05, 0.1, 0.2],
    "gamma": [0.01, 0.05, 0.5, 1],
    "kernel": ["rbf", "poly", "sigmoid"]
}

from sklearn.model_selection import ParameterGrid
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
best_f1 = -1
best_params = None
for params in ParameterGrid(param_grid):
  model = OneClassSVM(**params)
  model.fit(X_train)
  y_pred_train = model.predict(X_train)
  y_pred_outliers = model.predict(X_outliers)
  y_true = np.hstack([np.ones(len(X_train)), -1 * np.ones(len(X_outliers))])
  y_pred = np.hstack([y_pred_train, y_pred_outliers])
  f1 = f1_score(y_true, y_pred, pos_label = -1)
  if f1 > best_f1:
    best_f1 = f1
    best_params = params
print(f"최적의 하이퍼파리미터: {best_params}")
print(f"최고 F1-score: {best_f1 : .4f}")

# load_breast_cancer 데이터에 대해 nuSVC 모델을 적용하시오.
# - 정규화해서 진행하시오.
# - PCA를 이용해서 최적의 차원으로 차원 축소하시오.
# - f1 score를 이용해서 평가하시오.
# - RandomizedSearchCV로 최적의 파라미터를 튜닝하시오.
# - 파일로 제출(날짜_주제_이름.ipynb) , 수업 끝나기 전까지