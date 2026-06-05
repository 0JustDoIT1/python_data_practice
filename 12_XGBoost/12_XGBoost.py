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

# Dmatrix(data matrix)
# 메모리를 효율적으로 사용(희소행렬에 대해서)
# 고속연산 지원

import xgboost as xgb
xgb.__version__

config = xgb.get_config()
print(config) # nthread 병렬처리, use_cuda_async_pool 쿠다지원

data = [[1,2,3], [4,5,6], [7,8,9]]
label = [1,0,1] # 1x3
dmatrix = xgb.DMatrix(data, label) # 독립변수 + 종속변수 함께 다룸

# 내부적으로 희소행렬 중심
label = dmatrix.get_label() # 함수이용처리
print(label)

data = dmatrix.get_data()
print(data)
dmatrix.save_binary("data.dmatrix") # 2진저장
dmatrix = xgb.DMatrix("data.dmatrix") # 인스턴스하면서 로딩

# dataframe: loc, iloc으로 인덱싱
sliced_dmatrix = dmatrix.slice([0,1],[0,1])
print(sliced_dmatrix)
print(sliced_dmatrix.get_data())

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data=load_iris() # Bunch data
X_train, X_test, y_train, y_test = train_test_split(data["data"],data["target"], test_size =.2, random_state = 42)

# DMatrix : DataFrame을 이용해서 초기화 가능
# feature_names를 별도로 지정
dtrain = xgb.DMatrix(data =X_train, label=y_train, feature_names = data.feature_names)
dtest = xgb.DMatrix(data =X_test, label=y_test, feature_names = data.feature_names)
# 규제: max_depth
# eta: learning-rate
# objective 목적함수 : 최적화의 대상
# softmax 다중분류 : 모델이 출력한 데이터들의 확률값으로 변환
# exponent로 작고 큰값을 확실하게 구분

param = {"max_depth":2, "eta":1, "objective":"multi:softmax", "num_class":3, "eval_metric": "mlogloss"} # mlogloss: 10진수
num_round = 2 # 모델 구성 횟수
# Dmatrisx를 이용한 모델 학습
# train/cv (cross validation)
bst = xgb.train(param, dtrain, num_round) # 리턴이 모델
preds = bst.predict(dtest) # softmax가 적용이 된 상태 최고 확률값의 인덱스 출력
print(preds[:5])

# output_margin = True -> softmax 입력되기 전의 계산된 결과값
import numpy as np

# softmax 적용전 값
y_raw_scores = bst.predict(dtest, output_margin=True)
# 여러개의 값이 입력
def softmax(x): # 지수함수 -> 작은 값은 더 작게, 큰 값은 더 크게 확인
  e_x = np.exp(x- np.max(x, axis=1, keepdims =True))
  return e_x / np.sum(e_x, axis =1, keepdims =True)
y_pred_proba_manual = softmax(y_raw_scores)

print("Raw Scores")
print(y_raw_scores[:5].round(4))

print("\nSoftmax Probabilities:")
print(y_pred_proba_manual[:5].round(4))


evals_result = {}
eval_list = [(dtrain, "train"), (dtest, "eval")]
bst = xgb.train(
    param,
    dtrain,
    num_round,
    evals = eval_list, # 평가데이터
    evals_result = evals_result, # 평가 결과
    verbose_eval = False # 출력 여부 / true 선택시 학습하면서 계속 출력함
)
preds = bst. predict(dtest)
print(f"예측 결과(상위 5개): {preds[:5]}")

# 시각화1
print("3.1. 특성 중요도 시각화(Feature Importance)")
fig, ax = plt.subplots(figsize=(10,5))
xgb.plot_importance(bst, ax=ax, importance_type = "gain", title="Feature Importance(Gain)")
plt.show()

# 시각화2
# num_round =2 이기 때문에 Tree가 2개 작성
print("3.2.개별 트리 시각화(Tree Structure)")
fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_tree(bst, num_tree=1, ax=ax)
plt.title("XGBoost Tree", fontsize=16)
plt.show()

# evals_result 학습하면서 평가한 결과
# 데이터 1벌 전체 학습한 횟수 -> 경사하강법으로 최적화
epochs = len(evals_result["train"]["mlogloss"])
x_axis = range(0, epochs)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x_axis, evals_result["train"]["mlogloss"],
        label = "Train Log Loss")
ax.plot(x_axis, evals_result["eval"]["mlogloss"],
        label = "Eval Log Loss")
ax.legend()

plt.ylabel("Log Loss")
plt.xlabel("Boosting Rounds")
plt.title("XGBoost Learning Curve")
plt.show()

# 현재 사용하고 있는 부스터 확인
# 부스터 종류(3가지) : gblinear, gbtree, dart
import json
config = json.loads(bst.save_config()) # 문자 -> json변환
booster_type = config["learner"]["gradient_booster"]["name"]
print(f"현재 부스터 종류: {booster_type}")

# booster를 dart로 학습
# dart는 dropout을 지원

param = {"booster" : "dart",           # dropout 지원
         "max_depth": 5,               # 규제
         "learning_rate": 0.1,         # 학습률을 작게 주면 -> 많은 트리
         "objective": "multi:softmax", # 다중분류
         "num_class": 3,
         "sample_type": "uniform",     # 트리를 없앨때 균등
         "normalize_type": "tree",     # 정규화 트리값으로
         "rate_drop": 0.1, # 기존에 만들어진 트리에서 얼마나 제거할 것인지(10%)
         "skip_drop": 0.5  # 50% 즉 2번 중에 1번만 실행
         }

num_round = 50
bst = xgb.train(param, dtrain, num_round)
preds =bst.predict(dtest)
preds

# DMatrix 모델을 wrapper한 XGBClassifier
# scikits와 DMAtrix를 연결하는 모델
# Pipeline, parameter tunning을 지원

from xgboost import XGBClassifier
# DT, RF 사용하던 파라미터 사용
xcl = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, num_class=3, objective = "multi:softmax")
xcl.fit(X_train, y_train)
preds = xcl.predict(X_test)
xcl.score(X_test, y_test) # scikits함수

print(xcl.get_booster())
xcl # 파라미터 확인
# booster = None

xcl.get_params()["booster"]
# 없어서 결과가 안나옴

model = xgb.XGBClassifier(booster="dart")
print(model.get_params()["booster"])
# 지정해줬을 때 결과 나옴

# 간이 문제
# accuracy_score를 이용해서 평가해보시오 -> metrics에 있는 정도 암기
from sklearn.metrics import accuracy_score # 일반화 함수
# 예측한 값, 실제값이 들어감
y_pred_labels = xcl.predict(X_test)
print(y_pred_labels[:5])
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\n테스트 데이터 정확도: {accuracy:.4f}")





# 확률로 출력
y_pred_proba = xcl.predict_proba(X_test) # 확률값
print(f"확률 예측(predict_proba 출력):[클래스 0, 클래스 1, 클래스 2]")
print(y_pred_proba[:5].round(4))
print(f"\n첫 번째 샘플의 예측 레이블(argmax): {np.argmax(y_pred_proba[0])}")

# dart로 학습

# DMatrix의 dart 부스터와 XGBClassifier의 파라미터가 동일함
# EvaluationMonitor, LearningrateSchefuler, TrainingCheckpoint가 새로 들어옴 / 이전에는 EarlyStopping만
model = xgb.XGBClassifier(n_estimators =50,
                          booster = "dart",
                          max_depth = 5,
                          learning_rate = 0.1,
                          objective = "multi:softmax",
                          num_class=3,
                          sampel_type = "uniform",
                          normalize_type = "tree",
                          rate_drop = 0.1,
                          skip_drop = 0.5)
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.score(X_test, y_test)

# early-stopping 적용
# fitting에다 callback함수를 적용(옛날) -> 모델에서 적용(현재)

# keras 문법 -> XGBoost 적용
# EarlyStopping : 개선이 없으면 학습 중지
custom_early_stop = xgb.callback.EarlyStopping(
    rounds = 20, # mlogloss의 개선이 20회까지는 참아라
    save_best = True, # 가장 좋은 모델을 저장
    maximize = False,
    data_name = "validation_1",
    metric_name = "mlogloss"# 평가기준
)

model = xgb.XGBClassifier(
    # gradient boost는 순차모델 (잔차 개선 - 앞의 모델과 연관)
    # randomforest는 독립모델 (모델끼리 독립)
    n_estimators = 500, # round 횟수 - dmatrix의
    booster = "dart", # dropout
    max_depth = 5,
    learning_rate = 0.1,
    objective = "multi:softprob",
    num_class = 3,
    sample_type = "uniform", # 트리를 제거하는 확률(weighted)
    normalize_type = "tree",
    rate_drop = 0.1,
    skip_drop = 0.5, # 2번중에 한번
    eval_metric = ["mlogloss", "merror"],
    random_state = 42,
    callbacks = [custom_early_stop] # 모델에 적용, 여러개 적용가능
)

model.fit(
    X_train, y_train,
    eval_set = [(X_train, y_train), (X_test, y_test)],
    verbose = True
)

results = model.evals_result()
plt.figure(figsize=(8,5))
plt.plot(results["validation_0"]["mlogloss"],
         label="train mlogloss")
plt.plot(results["validation_1"]["mlogloss"],
         label="test mlogloss")
plt.xlabel("Boosting rounds")
plt.ylabel("mlogloss")
plt.legend()
plt.title("Learning Curve(mlogloss)")
plt.gird =True
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
num_classes = 3
cm = confusion_matrix(y_test, preds, labels=np.arange(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.arange(num_classes))
disp.plot(cmap = "Blues", values_format = "d")
plt.title("Confusion Matrix")
plt.show()
# 오분류 없음을 확인 (대각선 빼고 다 0)

# regression (회귀)

# regression
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
title = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "RAD","TAX", "PTRATIO", "B", "LSTAT", "medv"]
raw_df = pd.read_csv(data_url, sep= "\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :],
                  raw_df.values[1::2, :2]])

target = raw_df.values[1::2, 2]
print(type(data))
boston_df = pd.DataFrame(data, columns = title)
boston_df.head()

X = boston_df.iloc[:, :12] # 독립변수
print(X.shape)
y = boston_df.iloc[:, 12] # 종속변수
y.shape

from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# 회귀 규제
# alpha(L1), lambda(L2), alpha + lambda : elasticnet
# colsample_bytree(열샘플) : subsample(행샘플링)
# early-stopping
# gblinear, gbtree, dart 전부 -> regression + classification

xg_reg = xgb.XGBRegressor(booster = "gbtree", objective = "reg:squarederror", colsample_bytreee = 0.3, learning_rate = 0.1, max_depth=5, alpha = 10, n_estimators = 20) # MSE
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# !apt-get -qq install -y graphviz && pip install -q pydot
# !apt-get install graphviz libgraphviz-dev pkg-config
# !pip install pygraphviz

# 시각화
import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg, num_trees=3) # tree수를 1,2,3 넣어보기(점점 개선)
plt.rcParams["figure.figsize"] = [15, 7.5]
plt.show()

# 위의 데이터에 DMatrix 적용한 모델 학습
data_dmatrix = xgb.DMatrix(data = X, label = y)
params = {"objective": "reg:squarederror",
          "colsample_bytree" : 0.3,
          "learning_rate" : 0.1,
          "max_depth":5,
          "alpha" :10,
          "n_jobs" : -1} # 문제 때문에 추가 -> "n_jobs" : -1

cv_results = xgb.cv(dtrain = data_dmatrix, params = params,
                    nfold = 3, # 데이터 분할 개수
                    num_boost_round = 50,
                    early_stopping_rounds = 10,
                    metrics = "rmse",
                    as_pandas = True, # 결과값을 pandas로 리턴
                    seed = 123,
                    # cv 표준편차
                    # cross_val_score => mean(), std()
                    callbacks = [xgb.callback.EvaluationMonitor(show_stdv=True)])
print(cv_results.head())

# 교차검증(cv) 결과에서 가장 성능이 좋았던 트리 개수(round)를 찾는 코드
best_round = cv_results["test-rmse-mean"].idxmin()
best_round # 계속 개선이 진행

# 결과 :1.5380 -> 과적합 의미
last_row = cv_results.iloc[-1]
gap = last_row["test-rmse-mean"] - last_row["train-rmse-mean"]
gap

# 그래프 시각화
plt. figure(figsize = (10,6))
plt.plot(cv_results["train-rmse-mean"], label = "Train RMSE")
plt.plot(cv_results["test-rmse-mean"], label = "Test RMSE")

plt.xlabel("Boosting Rounds")
plt.ylabel("RMSE")
plt.title("XGBoost CV Learning Curve")
plt.legend()
plt.grid = True
plt.show()

# hyper paramter tunning
import scipy.stats as st
params = {
    "n_estimators" : st.randint(3,40),
    "max_depth" : st.randint(3,40),
    "learning_rate": st.uniform(0.05, 4),
    "gamma" : st.uniform(0,10)
}

# RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(xg_reg, params, n_jobs = -1)
gs.fit(X_train, y_train)
gs.best_estimator_

# 문제
# RandomizedSearchCV 에서 n_jobs =1을 주었을 때와 안주었을 때의 시간 차
# xgb.cv n_jobs = -1 때의 시간 차
# RandmizedSearchCV에서 n_jobs = -1 을 지정
# xgb.cv 나 xgb.XGBRegressor에 n_jobs = -1을 지정

# 4가지 경우의 실행을 비교해서 출력하시오.
# 상승 작용있음 (고속진행)
# %% time 이용

gs.best_params_["gamma"]

# DMatrix -> Dataframe

from sklearn.preprocessing import LabelEncoder

fillmodel = xgb.XGBClassifier(
    random_state=42,
    gamma=gs.best_params_["gamma"],
    learning_rate=gs.best_params_["learning_rate"],
    max_depth=gs.best_params_["max_depth"],
    n_estimators=gs.best_params_["n_estimators"]
)

# DMatrix -> Dataframe
X_features = data_dmatrix.get_data()
if hasattr(X_features, "toarray"):
  X_features = X_features.toarray()
df_back = pd.DataFrame(X_features, columns=data_dmatrix.feature_names)
le = LabelEncoder() # 0,1,2 숫자 변환
y_labels_encoded = le.fit_transform(data_dmatrix.get_label())

fillmodel.fit(df_back, y_labels_encoded)
y_pred = fillmodel.predict(df_back)
accuracy = accuracy_score(y_labels_encoded, y_pred)
print(f"{accuracy:.4f}")

# pima.csv 를 이용한 모델화
import pandas as pd
dataset = pd.read_csv("/content/drive/MyDrive/[미래융합교육원] AI빅데이터_파이썬/dataset/pima.csv")
print(dataset.shape)
dataset.head()

# 전처리 : 결측치(자동), 이상치(트리자체둔감), 범주화(트리 자체가 범주화), 정규화(사이즈에 영향을 받지 않음)
X = dataset.iloc[:, 1:dataset.shape[1]-1] # 독립변수
print(X.shape)
print(X.head())
y = dataset.iloc[:, dataset.shape[1]-1] # 종속변수 하나만 가져옴
# 범주형
y = np.where(y == 'Yes', 1, 0)
y = pd.Series(y)

# 문제
# test_train을 분리
from sklearn.model_selection import train_test_split
test_size = 0.33
# random_state -> 모델 결과가 같은 결과가 되도록 정함
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)

model = xgb.XGBClassifier(booster="dart", n_jobs = -1, learning_rate = 0.1, max_depth = 3, random_state = 200)
model.fit(X_train, y_train)
print(model)

# 문제 accuracy_score를 이용해서 train 정확도와 test 정확도를 출력
from sklearn.model_selection import cross_val_score

y_pred_train = model.predict(X_train)
print(y_pred_train)
accuracy = accuracy_score(y_train, y_pred_train)
print("훈련정확도: %.2f%%" % (accuracy * 100.0))
model.score(X_test, y_test)
y_pred = model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("테스트정확도: %.2f%%" % (accuracy * 100.0))
# 훈련정확도: 97.30%, 테스트정확도: 72.73%
# 과적합

result = cross_val_score(model, X_test, y_test)
result.mean()

result.std()

# tree 모델의 변수 중요도를 이용한 변수선택법
from sklearn.feature_selection import SelectFromModel
print(model.feature_importances_)
thresholds = np.sort(model.feature_importances_)
print(thresholds)
print(np.sum(thresholds))

import warnings
warnings.filterwarnings('ignore')
for thresh in thresholds: # 변수 중요도
  # 변수 제거
  # prefit=True : 변수 중요도를 미리 학습한 것을 사용
  # (false라면 디폴트 파라미터를 이용해서 변수 중요도 계산)
  selection = SelectFromModel(model, threshold=thresh, prefit=True)
  # 차원축소 (변수 선택으로)
  select_X_train = selection.transform(X_train) # 변수를 제거
  # use_label_encoder 범주형의 one-hot-encoder를 자동
  selection_model = xgb.XGBClassifier(use_label_encoder=False)
  selection_model.fit(select_X_train, y_train)
  # train 데이터에서 변수 제거, test 데이터에서도 변수 제거
  select_X_test = selection.transform(X_test)
  y_pred = selection_model.predict(select_X_test)
  predictions = [round(value) for value in y_pred]
  accuracy = accuracy_score(y_test, predictions)
  # select_X_train -> 제거하고 남은 변수의 개수
  print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

X_train.shape
select_X_train.shape

thresh=0.092
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
selection_model = xgb.XGBClassifier(use_label_encoder=False)
selection_model.fit(select_X_train, y_train)
select_X_test = selection.transform(X_test)
y_pred = selection_model.predict(select_X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
params = {
    "n_estimators" : st.randint(3,40),
    "max_depth" : st.randint(3,40),
    "learning_rate": st.uniform(0.05, 4),
    "gamma" : st.uniform(0,10)
}
gs = RandomizedSearchCV(selection_model, params, n_jobs=-1, verbose=0)
gs.fit(select_X_train, y_train)
gs.best_params_

# 훈련정확도는 내려야 되고, test 정확도는 올려서 일반화
y_pred_train = gs.best_estimator_.predict(select_X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print("훈련정확도: %.2f%%" % (accuracy * 100.0))

import joblib
joblib.dump(gs.best_estimator_, "pima.joblib.dat")
loaded_model = joblib.load("pima.joblib.dat")
y_pred = loaded_model.predict(select_X_test)
accuracy = accuracy_score(y_test, y_pred)
print("훈련정확도: %.2f%%" % (accuracy * 100.0))
# balancing : train 데이터 과적합

# 데이터 불균형을 고려한 모델
# negative vs positive
neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / max(pos, 1)
print(f"[Class balance] neg={neg}, pos={pos}, scale_pos_weight≈{scale_pos_weight:.2f}")

common_kwargs = dict(
    n_estimators = 400,             # 모델의 개수
    max_depth = 3,                  # 모델의 깊이
    learning_rate = 0.05,           # 학습률
    subsample = 0.9,                # 행 샘플링
    colsample_bytree = 0.8,         # 열 샘플링
    reg_lambda=1.0,                 # 규제
    objective = 'binary:logistic',  # 이진분류
    eval_metric = 'logloss',        #
    tree_method = 'hist',           # histogram은 연속적 데이터 범주화
    random_state=42,                #
    n_jobs = -1
)

xgb_base = XGBClassifier(**common_kwargs)
xgb_base.fit(select_X_train, y_train, eval_set = [(select_X_test, y_test)], verbose =False)

# 데이터 불균형 문제
# scale_pos_weight => 불균형 가중치
xgb_spw = XGBClassifier (**common_kwargs, scale_pos_weight=scale_pos_weight)
xgb_spw.fit(select_X_train, y_train, eval_set=[(select_X_test, y_test)], verbose=False )

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, RocCurveDisplay,
    PrecisionRecallDisplay, average_precision_score
)

def evaluate(model, X, y, name, thr=0.5):
  proba = model.predict_proba(X)[:, 1]
  pred = (proba >= thr).astype(int)
  res = {
      "Model": name,
      "AUC" : roc_auc_score(y, proba),
      "AP (PR-AUC)": average_precision_score(y, proba),
      "Accuracy": accuracy_score(y, pred),
      "Precision": precision_score(y, pred, zero_division=0),
      "Recall": recall_score(y, pred, zero_division=0),
      "F1": f1_score(y, pred, zero_division=0),
      "ConfusionMatrix (TN,FP,FN,TP)": tuple(confusion_matrix(y, pred).ravel())
  }
  return res, proba

res_base, proba_base = evaluate(xgb_base, select_X_test, y_test, "XGB (no weight)")
res_spw, proba_spw = evaluate(xgb_spw, select_X_test, y_test, f"XGB (scale_pos_weight={scale_pos_weight:.2f})")
print(pd.DataFrame([res_base, res_spw]).round(4))
# scale_pos_weight 고려한 모델 : 가중치를 주지 않은 것보다 좋아짐
# 일반적으로 다 좋아지진 않고, recall이 증가하고 Accuracy는 떨어짐
# Precision : positive로 예측했는데 정확하게 맞힌 것
# Recall : 진짜 양성을 얼마나 잘 맞추었는가
# bias vs variance 관계도 trade off 관계
# Precision vs recall : trade off 관계가 있음 -> 하나가 좋아지면 하나가 나빠짐
# threshold를 조절 : 확률에서
# logistic model : 0.5 (threshold)
# recall이 증감 : 이 때 최적의 accuracy를 얻는 것이 목적

# 임계값 tunning
# accuracy 도 상승 -> recall 이 상승 (환자를 환자로 예측)
# f1 = recall / precision 을 둘 다 고려하는 평가 지표
def best_threshold_by_f1(model, Xv, yv): # 모델, 독립변수/종속변수
  p = model.predict_proba(Xv)[:,1] # 확률로 예측
  thrs = np.linspace(0.05, 0.95, 19) # 경계값을 생성
  scores = [(thr, f1_score(yv, (p >= thr).astype(int))) for thr in thrs]
  return max(scores, key=lambda x: x[1])
# 기준값은 0.5
# base thr=0.15, F1=0.6437
# spw thr=0.45, F1=0.6750
thr_base, f1b = best_threshold_by_f1(xgb_base, select_X_test, y_test)
thr_spw, f1s = best_threshold_by_f1(xgb_spw, select_X_test, y_test)
print(f"[Best F1] base thr={thr_base:.2f}, F1={f1b:.4f} | spw thr={thr_spw:.2f}, F1={f1s:.4f}")