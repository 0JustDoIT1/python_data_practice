# 이산적 naive bayes
# Bayes 원리
# 사후확률 = (사전확률 * 우도) / 증거
# 증거 = 각 데이터의 우도의 합
# 지도학습 : labeling -> 도수분포표 -> 사전 확률
# 데이터로부터 우도 (메일에서 메일에 들어 있는 단어들)
# spam 들어가는 단어
# pam 들어가는 단어
# spam과 pam의 확률을 각기 구함
# 증거는 spam우도 + pam우도

# 시험은 scikits-learn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # 분류

# 다범주, true/false 범주
from sklearn.naive_bayes import CategoricalNB, BernoulliNB


data = pd.DataFrame({
    "Outlook": [ # 맑음 / 흐림 / 비
        "Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain",
        "Overcast", "Sunny", "Sunny", "Rain", "Sunny",
        "Overcast", "Overcast", "Rain"
    ],
    "Temperature": [ # 온도: 더움 / 온화 / 서늘
        "Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
        "Cool", "Mild", "Cool", "Mild", "Mild",
        "Mild", "Hot", "Mild"
    ],
    "Humidity": [ # 습도 : 높음 / 정상
        "High", "High", "High", "High", "Normal", "Normal",
        "Normal", "High", "Normal", "Normal", "Normal",
        "High", "Normal", "High"
    ],
    "Windy": [ # 바람의 여부
        False, True, False, False, False, True,
        True, False, False, False, True,
        True, False, True
    ],
    "Play": [ # 운동 : 할거냐 말거냐
        "No", "No", "Yes", "Yes", "Yes", "No",
        "Yes", "No", "Yes", "Yes", "Yes",
        "Yes", "Yes", "No"
    ]
})

# 변수가 5개


X = data.drop("Play", axis = 1) # 독립변수
y = data["Play"] # 종속변수
print(data)

print(y.value_counts()) # 사전확률
# Play
# Yes    9
# No     5
# Name: count, dtype: int64

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify  = y)

# Categorical NB
# 기본적으로 범주형 인덱스를 사용

cat_nb_pipe = Pipeline([
    ("encoder", OrdinalEncoder()), # 범주형 변수
    ("model", CategoricalNB())

])
cat_nb_pipe.fit(X_train, y_train)
y_pred_cat = cat_nb_pipe.predict(X_test)
y_prob_cat = cat_nb_pipe.predict_proba(X_test)
print("CategofricaNB 정확도 :", accuracy_score(y_test, y_pred_cat))

print(confusion_matrix(y_test, y_pred_cat))
print(classification_report(y_test, y_pred_cat))

# [[0 2]
#  [0 3]]

#  대각선에 있는 것은 : 정분류 (0,3)
#  나머지 : 오분류(0,2)

# 베루누이NB
# true / false 모델
# ONeHotEncoder : 데이터가 있는지 없는지(0,1)

ber_nb_pipe = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),("model", BernoulliNB())
])

# 증거를 확인하고 확률 -> 사후 확률로 바꿈 -> 큰 확률값을 선택
ber_nb_pipe.fit(X_train, y_train)

y_pred_ber = ber_nb_pipe.predict(X_test)
y_prob_ber = ber_nb_pipe.predict_proba(X_test)

print("BernoulliNB 정확도:", accuracy_score(y_test, y_pred_ber))
print(confusion_matrix(y_test, y_pred_ber))
print(classification_report(y_test, y_pred_ber))

# 제일 중요한 hyper parameter : alpha - smoothing
# true/false 0이 많아서 확률 계산 시 0으로 가는 것을 막아줌
# 스무딩을 키워주면 확률이 모두 같게 되는 현상

alpha_grid =  {
    "model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
}
cv = StratifiedKFold( #  층화 분휴 *(y분포가 원래 데이터와 비슷하게)
    n_splits = 3,
    shuffle = True,
    random_state = 42
)
cat_grid = GridSearchCV(
    estimator = cat_nb_pipe,
    param_grid = alpha_grid, # smoothing 조절 파라미터
    scoring ="accuracy",
    cv= cv, # 개수
    return_train_score = True
)

cat_grid.fit(X_train, y_train)

cat_grid.best_params_
best_cat = cat_grid.best_estimator_

best_cat.predict(X_test)

# 신규 데이터 예측

new_weather = pd.DataFrame({
    "Outlook": ["Sunny"],
    "Temperature": ["Hot"],
    "Humidity": ["High"],
    "Windy": [False]})

best_cat.predict(new_weather)

best_cat.predict_proba(new_weather)

# 실제 모델서비스를 하는 함수를 작성하시오
# 한번 하고 끝나는 것이 아닌, 요청이 들어오면 반복해야하는 함수를 작성하면 됨.

def cat_grid(new_weather):
  new_weather = pd.DataFrame(new_weather)
  for i in range(len(new_weather)):
    if new_weather["Outlook"][i] : 1
    new_weather["Outlook"][i] == 2
  return cat_grid.best_estimator_.predict_proba(new_weather)


# 함수 정의
# 웹에서 입력된 데이터
def predict_play_tennis(model, outlook, temperature, humidity, windy):
  sample = pd.DataFrame({
      "Outlook": [outlook], #변수
      "Temperature": [temperature],
      "Humidity": [ humidity],
      "Windy" : [windy]
  })


  pred = model.predict(sample)[0]
  prob = model.predict_proba(sample)[0]
  result = pd.DataFrame({
      "Class" : model.classes_,
      "Probability" : prob
  })
  return pred, result

  # 함수 호출
pred, prob_table = predict_play_tennis(best_cat, outlook= 'Rain', temperature= "Mild", humidity = "Normal",windy = False )

pred, prob_table

# Multinomial NB
# ML : 이미지 대량 텍스트, 모델은 배제
# text mode : 베트남. 인니에서 한국 파워 분석 (기사분석)
# 손글씨

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X_dg, y_dg = digits.data, digits.target

# 8X8 컬러는 0~16 까지 정수 (이미지는 양수)

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

fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize =(12,4))
fig.suptitle('Digits 데이터셋 - 처음 4개 샘플 시각화', fontsize = 13, fontweight ='bold' )

# 데이터가 하나의 열이나 행으로 구성 64 일렬

for i in range(5) :
  ax= axes[i]
  image_8x8 = X_dg[i].reshape(8,8)
  # 보간법으로 대체
  ax.imshow(image_8x8, cmap = 'gray_r', interpolation = 'nearest') # 가장가까운 데이터로 대체

  ax.set_title(f'Index: {i}\nLabel(y): {y_dg[i]}', fontsize = 11)
  ax.axis('off')

plt.tight_layout()
plt.show()

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


X_tr_dg, X_te_dg, y_tr_dg, y_te_dg = train_test_split( X_dg, y_dg, test_size = 0.2, random_state = 42, stratify = y_dg)

# gridSearchCV 하지 않고 cross_val_score로 테스트

alphas = [0.001, 0.01,0.1, 0.5, 1.0, 2.0,5.0, 10.0]

# 최적의 alpha값 선정
# cv이 중요한 신뢰구간값을 제공

alpha_accs = [cross_val_score(MultinomialNB(alpha= a), X_dg, y_dg,cv=5).mean()
for a in alphas]

best_alpha_dg = alphas[np.argmax(alpha_accs)] # 가장 좋은 알파 선택
mnb_dg = MultinomialNB(alpha= best_alpha_dg)
mnb_dg.fit(X_tr_dg, y_tr_dg)
y_pred_dg = mnb_dg.predict(X_te_dg)




print(classification_report(y_te_dg, y_pred_dg))

# 문제 : 분산값은 얼마인지 출력하시오

alpha_accs = [cross_val_score(MultinomialNB(alpha=a), X_dg,y_dg, cv=5).std()for a in alphas]

print(alpha_accs)

# 시각화

import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import learning_curve
fig, axes = plt.subplots(2, 3, figsize = (18, 10))
fig.suptitle('MultinnomialNB', fontsize = 14, fontweight = 'bold')
plt.close(fig)
# 2 x 3 6개의 ax


C = {'blue':'#2E86AB','red':'#E74C3C','green':'#27AE60',
     'orange':'#E67E22','purple':'#8E44AD','teal':'#00897B'}

# 셀마다 log 확률을 계산한 값 출력
# class_log_prior_ 사전확률: 값이 많이 크거나, 많이 작거나 할 때 log를 붙임.
# feature_log_prob_ 우도 : 64개 -> 픽셀 1:1 대응 -> 이미지 출력 가능
# 이미지 사이즈 64바이트로 구성됨 -> 입력변수는 몇개?
# 픽셀 한개 한개가 입력값
# 400 x 400 입력 데이터는 몇 차원 :  160000 차원 데이터
# 픽셀 1개를 입력 데이터로 봄

# 꼭 알아야 할 것
# 이미지는 하나의 픽셀이 하나의 입력이다
# 변수가 64개


ax = axes[0, 0]
for subplot_pos, digit in enumerate([0, 4, 9]):
  inner_ax = fig.add_axes([0.05 + subplot_pos * 0.10, 0.56, 0.09, 0.14])
  # feature_log_prob_ : 64개 구성
  log_prob = mnb_dg.feature_log_prob_[digit].reshape(8, 8)
  inner_ax.imshow(log_prob, cmap = 'Blues', interpolation = 'nearest')
  inner_ax.set_title(f'숫자 {digit}', fontsize = 9, pad = 2)
  inner_ax.axis('off')
# 설명
ax.text(0.5, 0.5, '각 픽셀의 클래스별 로그 확률',
        ha = 'center',
        va = 'center',
        fontsize = 11,
        transform = ax.transAxes,
        bbox = dict(boxstyle = 'round', facecolor = '#EEF5FD', edgecolor = '#2E86AB'))
ax.axis('off')
ax.set_title('학습된 특징 로그확률', fontsize = 11)

fig


# x축값 간격 비균등할 때 log값을 취해서 표현
ax = axes[0, 1]
ax.semilogx(alphas, alpha_accs, 's-', color = C['purple'], lw = 2.5, ms = 8)
ax.axvline(best_alpha_dg, color = C['red'], ls = '--', lw = 2, label = f'최적 α = {best_alpha_dg}')
# layer 계층 맨 위로 출력
ax.scatter([best_alpha_dg], [max(alpha_accs)], color = C['red'], s = 150, zorder = 5)
ax.set_xlabel('스무딩 파라미터 α (라플라스)' )
ax.set_ylabel('5-fold CV 정확도')
ax.set_title('α 라플라스 스무딩 튜닝', fontsize = 11)
ax.legend(fontsize = 9)
fig

ax= axes[0,2]
cm_dg = confusion_matrix(y_te_dg, y_pred_dg)
sns.heatmap(cm_dg, annot=True, fmt = 'd', cmap = 'Blues', ax= ax, cbar = False , annot_kws = {'size': 8})

ax.set_title('혼동행렬(0-9)', fontsize = 11)
ax.set_xlabel('예측'); ax.set_ylabel('실제')

fig

# 오분류된 샘플 출력
# 이미지를 출력해서 틀린 이유를 확인

ax = axes[1,0]
wrong_idx = np.where(y_pred_dg !=y_te_dg)[0][:12]
if len(wrong_idx) > 0:
  ax.axis('off') # 그림만 출력

  ax.set_title(f"오분류 샘플(상위 {min(12, len(wrong_idx))}개)", fontsize = 11)
  n_cols = 6 # 열이 6개 2x6으로 출력
  # 순번, 인덱스
  for idx_i, wi in enumerate(wrong_idx[:12]):
    # 출력위치 결정(인덱스)
    row_i = idx_i //n_cols # 몫연산자 0,1
    col_i = idx_i % n_cols # 좌우 순서
    # 축을 추가해서 화면을 분할
    # 좌측 시작점, 하단 시작점, 가로폭, 세로폭

    inner = fig.add_axes ([0.06 + col_i * 0.058, 0.10 + (1-row_i)* 0.125 , 0.058, 0.11])
    inner.imshow(X_te_dg[wi].reshape(8,8),
                 cmap ='gray_r', interpolation = 'nearest')
    inner.set_title(f'실제: {y_te_dg[wi]}\n예측:{y_pred_dg[wi]}', fontsize = 6.5, color = C['red'], pad = 1)
    inner.axis('off')
else :
  ax.text(0.5, 0.5, '오분류 샘플이 없습니다.', ha= 'center', va= 'center')
  ax.set_title('오분류 샘플 시각화', fontsize = 11)
fig

ax = axes[1,1]
prec = precision_score(y_te_dg, y_pred_dg, average = None)
rec = recall_score(y_te_dg, y_pred_dg, average = None)
x_pos = np.arange(10)
w = 0.35
# 한자리에 2개를 쌍으로 출력
# 원래 barplot 하게 되면 자기 위치의 중앙에 출력
# 재현율이 의미하는 것: 첫번째 열 = 진짜를 얼마나 잘 맞추었나
# 정밀도 : positive 하게 예측했는데 얼마나 맞췄는가
# 둘은 trade-off 관계에 있다-> 균형점을 찾아서 (조금씩 양보해서)
ax.bar (x_pos-w/2, prec, w, label ='정밀도', color = C['blue'], edgecolor = 'black', alpha = 0.85)
ax.bar(x_pos+w/2, rec, w, label ='재현율', color = C['orange'], edgecolor = 'black', alpha = 0.85)
ax.set_xticks(x_pos)
ax.set_xlabel('숫자 클래스')
ax.set_title('클래스별 정밀도 , 재현율', fontsize = 11)
ax.legend(fontsize =9)
ax.set_ylim(0, 1.1)
fig

ax = axes[ 1,2]
# tr_s 의 사이즈
# 5 x 10
# time series
ts, tr_s, vl_s = learning_curve(MultinomialNB(alpha = best_alpha_dg), X_dg, y_dg, train_sizes= np.linspace(0.1, 1, 10))
ax.plot(ts, tr_s.mean(1), 'o-', color = C['blue'], label='학습')
ax.fill_between(ts, tr_s.mean(1) - tr_s.std(1), # 열 기준
                tr_s.mean(1) + tr_s.std(1),
                alpha= 0.15, color = C['blue'])
ax.plot(ts, vl_s.mean(1), 's-', color= C['orange'], label = '검증')
ax.fill_between(ts, vl_s.mean(1)-vl_s.std(1),
                vl_s.mean(1) + vl_s.std(1),
                alpha= 0.05, color= C["orange"])
ax.set_xlabel('학습 샘플 수 ');ax.set_ylabel('정확도')
ax.set_title('학습곡선', fontsize = 11)

ax.legend(fontsize = 9)
fig.tight_layout()
fig

# 문제
# 와인 데이터
from sklearn.datasets import load_wine
wine = load_wine(as_frame=True)
X_wn, y_wn = wine.data, wine.target
cls_wn = wine.target_names
# MultinomialNB 모델을 구축하고
# 상위 3개의 특징의 클래스별 확률을 시각화(bar)하시오

# MultinomialNB # 분석대상 데이터 -> 이산적 데이터(도수)
# 도수의 특징 : 양수이면서 정수
# 데이터를 양수이면서 정수로 만드는 방법
# -> 절대값
# -> MinMaxScale(0 ~ 1) -> 양수로 변환 후 특정값 곱함
# 연속적 데이터 : gaussianNB

# Bunch에서 데이터 확인
wine = load_wine(as_frame=True) # 데이터 프레임으로 리턴
X_wn, y_wn = wine.data, wine.target
info_df = pd.DataFrame(X_wn.dtypes, columns=['Data Type'])
summary = X_wn.describe().T
summary

from sklearn.preprocessing import MinMaxScaler
scaler_wn = MinMaxScaler()
# MinMaxScaler는 양수를 만들고 -> 도수화 -> 정수
X_wn_scaled = (scaler_wn.fit_transform(X_wn) * 100).astype(int)
X_wn = X_wn_scaled
X_wn

# alpha 값 tunning
X_tr_wn, X_te_wn, y_tr_wn, y_te_wn = train_test_split(X_wn, y_wn, test_size = 0.25, random_state=42, stratify = y_wn)

alphas_wn = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

alpha_accs_wn = [cross_val_score(MultinomialNB(alpha = a),
                                 X_wn, y_wn, cv=5).mean() # 튜닝은 전체데이터
                                 for a in alphas_wn]
best_alpha_wn = alphas_wn[np.argmax(alpha_accs_wn)]
mnb_wn = MultinomialNB(alpha=best_alpha_wn)
mnb_wn.fit(X_tr_wn, y_tr_wn) # 학습은 학습 데이터로
y_pred_wn = mnb_wn.predict(X_te_wn)

accuracy_score(y_te_wn, y_pred_wn)

# 모델에서 중요한 변수를 선택하는 방법
# 우도값 : feature_log_prob_ # 데이터마다 변수의 우도값이 출력
# feature_log_prob_ 평균, 분산 중 어느 것 선택?
# -> 분산이 큰 변수가 중요한 변수
log_prob_diff = mnb_wn.feature_log_prob_.std(axis=0)
top5_feats = np.argsort(log_prob_diff)[::-1][:5]
top5_feats # 분산이 큰 변수

top3_feats = top5_feats[:3]

# 조건부확률 : 곱셈의 법칙
# 우도는 곱한 결과
# 0이 하나라도 있으면 결과 0
# log를 취해서 0으로 가지 않도록 확률을 log확률
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
fig.suptitle('MultinomialNB - 와인 클래스별 조건부 확률', fontsize=14, fontweight='bold', y=1.05)
colors = ['#800020', '#C71585', '#FFD700']
for i, fi in enumerate(top3_feats):
  ax = axes[i]
  feat_name = wine.feature_names[fi]
  # 0 ~ 1 사이의 확률값으로 복원
  # 우도값
  actual_probs = np.exp(mnb_wn.feature_log_prob_[:, fi])
  bars = ax.barh(cls_wn, actual_probs, color=colors,
                 edgecolor='black', alpha=0.85, height=0.5)
  # 우도값은 밀도값 (분포의 밀도값)
  for bar in bars: # bar가 한 특성마다 3개 출력
    width = bar.get_width() # 바의 오른쪽 길이
    # 텍스트 : 확률값 출력
    # 글씨를 조금 오른쪽
    ax.text(width + (max(actual_probs) * 0.02),
            bar.get_y() + bar.get_height()/2, # 위아래 중심출력
            f'{width:.4f}', ha='left', va='center',
            fontsize=10, fontweight='bold')
    ax.set_title(f'순위 {i + 1}: {feat_name}', fontsize=12,
                 pad=10, fontweight='bold')
    ax.set_xlabel('조건부 확률', fontsize=10)
    ax.set_ylabel('와인 품종', fontsize=10)
    ax.set_xlim(0, max(actual_probs) * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()