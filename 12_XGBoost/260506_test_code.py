# 문제
# diamonds = sns.load_dataset('diamonds')
# 에 대해서 다이아몬드 가격 회귀 모델을 구축하시오
# metric : rmse
# 회귀 성능 지표를 출력
# 잔차 분포를 출력하시오
# 변수 중요도를 출력하시오
# 변수중요도에 따른 변수 선택법을 적용해서 최적의 변수를 선택하고 모델을 개선하시오
# 로그변환한 모델과 원본 스케일 회귀모델을 비교하시오

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

diamonds = sns.load_dataset('diamonds')
diamonds.head()

X = diamonds.drop('price', axis=1) # 독립변수 : price 외에 컬럼 가져오기
y = diamonds['price'] # 종속변수 : price

# 기존 one-hot-encoding 진행하니까 변수가 너무 많이 늘어남
# OrdinalEncoder를 통해서 기존 변수를 유지하면서 범주형으로 지정
# 단, OrdinalEncoder에서는 순서가 중요하기 때문에 의미있게 순서 지정
encoder = OrdinalEncoder(categories=[
    ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],  # cut
    ['J', 'I', 'H', 'G', 'F', 'E', 'D'],               # color
    ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']  # clarity
])

X[['cut','color','clarity']] = encoder.fit_transform(
    X[['cut','color','clarity']]
)

# 훈련, 테스트 데이터 분리(테스트 0.3 기준)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

X.head()

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# XGBoost 회귀 모델
model = xgb.XGBRegressor(
    objective='reg:squarederror', # 모델이 무엇을 최소화하면서 학습할지 : MSE
    eval_metric='rmse', # 모델 성능 평가기준 : RMSE
    learning_rate=0.1, # 한 번에 얼마나 크게 업데이트할지
    max_depth=5, # 트리 깊이
    n_estimators=100, # 트리를 몇 개 만들지
    random_state=42 # 결과값 고정
  )

# 모델 학습
model.fit(X_train, y_train)

# 예측값 생성
y_pred = model.predict(X_test)
y_pred

# MSE 및 RMSE 계산
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse

# R2 계산(설명력)
r2 = r2_score(y_test, y_pred)
r2

print(f"(RMSE) : {rmse:.2f}")
print(f"(R2) : {r2:.4f}")

# (RMSE) : 535.90
# (R2) : 0.9816

import matplotlib.pyplot as plt

# 잔차 계산 (실제값 - 예측값)
residuals = y_test - y_pred
residuals

# 히스토그램으로 시각화
# bins=30 : 30개 구간으로 나눠서
# kde=True : 분포 곡선 추가
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residual Distribution")
plt.show()

# 변수 중요도 계산
# 계산한 값을 보기 좋게 DataFrame 형태로 변환
importance_df = pd.DataFrame({
    '변수': X.columns, # 각 컬럼명 매칭
    '중요도': model.feature_importances_ # 중요도
}).sort_values( # 중요도 순으로 정렬(내림차순)
    by='중요도',
    ascending=False
)

importance_df

import warnings
warnings.filterwarnings('ignore') # 불필요한 경고 무시

from sklearn.feature_selection import SelectFromModel

# 각 변수의 중요도 값을 오름차순으로 정렬해서 treshohold 후보로 사용
thresholds = np.sort(model.feature_importances_)

# 반복문 흐름
# 1. threshold 하나씩 적용
# 2. 해당 threshold 이상인 변수만 선택
# 3. 선택된 변수로 새 모델 학습
# 4. RMSE 측정 후 출력
for thresh in thresholds:
  selection = SelectFromModel(
      model, # 생성한 모델
      threshold=thresh, # 중요도가 thresh 이상인 변수만 남김
      prefit=True # 이미 학습된 모델 사용
  )
  # 선택된 변수만 남긴 새로운 train/test 데이터
  select_X_train = selection.transform(X_train)
  select_X_test = selection.transform(X_test)

  # 위에 기존 모델과 동일한 모델 사용
  selection_model = xgb.XGBRegressor(
      objective='reg:squarederror',
      eval_metric='rmse',
      enable_categorical=True,
      learning_rate=0.1,
      max_depth=5,
      n_estimators=100,
      random_state=42
  )
  # 모델 학습
  selection_model.fit(select_X_train, y_train)
  # 새로운 예측값
  y_pred = selection_model.predict(select_X_test)

  # RMSE 계산
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)

  print("Thresh=%.4f, n=%d, RMSE=%.2f" % (thresh, select_X_train.shape[1], rmse))

  # Thresh=0.0009, n=9, RMSE=535.90

# Thresh=0.0009, n=9, RMSE=535.90

# 최적 threshold 설정
best_thresh = 0.0009

# 변수 선택
selection = SelectFromModel(
    model,
    threshold=best_thresh, # 최적 thresh 적용(0.0009 이상인 변수만)
    prefit=True
)

# 선택된 변수 확인
selected_features = X.columns[ # 안에서 계산되어 나온 True / False 중에 True인 위치의 변수명만 추출
    selection.get_support() # 선택된 변수 위치를 True / False 배열로 반환
]

print("선택된 변수 개수:", len(selected_features))
print("선택된 변수:")
print(selected_features.tolist())

# 선택된 변수로 데이터 재구성 (train, test)
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# 개선 모델 생성 (동일하게)
improved_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    random_state=42
)

# 개선 모델 학습
improved_model.fit(X_train_sel, y_train)

# 예측값 생성
y_pred_sel = improved_model.predict(X_test_sel)

# 성능 평가
mse_sel = mean_squared_error(y_test, y_pred_sel)
rmse_sel = np.sqrt(mse_sel)

r2_sel = r2_score(y_test, y_pred_sel)

print(f"(RMSE) : {rmse_sel:.2f}")
print(f"(R2) : {r2_sel:.4f}")

# 변수 선택 이전
# (RMSE) : 535.90
# (R2) : 0.9816

# 변수 선택 이후
# (RMSE) : 536.72
# (R2) : 0.9815

# 결과 : 성능 차이가 거의 없음
# 변수선택 효과가 적은 이유 : XGBoost가 내부적으로 중요하지 않은 변수를 트리 분기에서 자연스럽게 무시
# 그럼에도 하는 이유 -> 성능이 비슷하다면 변수가 적은 모델이 유리 -> 속도 빠름, 해석 쉬움, 유지보수 쉬움

# 타겟 로그 변환
y_log = np.log1p(y) # log1p(x) -> 0이 있어도 안전하게 x+1로 로그 적용가능
# 로그 적용 이유 : price처럼 오른쪽으로 치우친 분포를 정규분포에 가깝게 변경

# 로그 데이터로 재분할(train, test)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.3, random_state=42
)

# 모델 학습
model.fit(X_train, y_train_log)

# 예측값 생성
pred_log = model.predict(X_test)

# 현재 예측값 및 테스트 데이터는 로그값 -> 역변환 -> 기존 스케일 복원
pred_price = np.expm1(pred_log)
actual_price = np.expm1(y_test_log)

# 성능평가
rmse_log = np.sqrt(mean_squared_error(actual_price, pred_price))
r2_log = r2_score(actual_price, pred_price)

print(f"(RMSE) : {rmse_log:.2f}")
print(f"(R2) : {r2_log:.4f}")

# 원스케일
# (RMSE) : 549.48
# (R2) : 0.9806

# 로그변환
# (RMSE) : 542.56
# (R2) : 0.9811

# 최종 : 로그 변환 모델이 소폭 우세