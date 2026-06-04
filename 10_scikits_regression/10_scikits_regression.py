import matplotlib.pyplot as plt # 시각화 script
# !apt-get update -qq # ubuntu 설치 명령어 (물어보지 말고 설치해라)
# !apt-get install fonts-nanum* -qq # 나눔 폰트 / 시스템에서 관리
import matplotlib.font_manager as fm # 글꼴 관리
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  name='NanumGothic')
# truetype font (t/f)
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'})

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1],[2],[3],[4],[5]]) # 독립변수 2차원 유지
y = np.array([2,4,6,8,10]) # 종속변수
model = LinearRegression() # default 파라미터
model.fit(X,y) # 학습
print("추정된 기울기(coef):", model.coef_)
print("추정된 절편(intercept):", model.intercept_)

X_new = np.array([[6],[7],[8]]) # 2차원 입력
y_pred = model.predict(X_new)

print("새로운 입력에 대한 예측값:" , y_pred)

# 시각화
plt.scatter(X, y, color="blue", label = "학습데이터")
plt.plot(X, model.predict(X), color="red", label="추정된 회귀선")
plt.scatter(X_new, y_pred, color="green", marker="x", s=100, label="새로운 예측값")

plt.title("Estimator vs Predictor", fontsize=16)
plt.xlabel("X 값", fontsize=12)
plt.ylabel("y값", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()

# 모델 불러오는 것 몇가지 정도 알아두기, 본인인 사용하는 모델이나
from sklearn import datasets, linear_model
# 평가함수들은 -> metrics에 있음
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y = True)
diabetes_X.shape

# train/test 분할
# test_size, train_size = 0.8, shuffle
# stratify 층화추출, y값 고려해서 나누어
X_train, X_test, y_train, y_test = train_test_split(
    diabetes_X, diabetes_y, random_state=0) # 75% vs 25% 자동분할
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
diabetes_y_pred = regr.predict(X_test)


print("Coefficients: \n", regr.coef_)
# 독립평가 함수: mean_squared_error 회귀 평가 지표
# regression vs classifier 모든 알고리즘
# 예측값 vs 실제값
print("Mean squared error: %.2f" % mean_squared_error(y_test, diabetes_y_pred))
print("결정계수: %.2f" % r2_score(y_test, diabetes_y_pred))
# r2_score == 모델.score 는 동일한 값
print(regr.score(X_test, y_test))

# Lasso(L1규제) , Ridge(L2 regularization), Elasticnet(L1,L2규제)
### L1 규제는 절대값으로 규제 - 계수가 너무 가파르면 과적합됨 -> 변수의 계수를 0으로 만들어서 변수를 선택하는 것이 특징

# Lasso
import seaborn as sns
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.datasets import load_diabetes
data = load_diabetes()
X = data.data
y = data.target
df = pd.DataFrame(data=np.c_[X,y], # numpy column 결합
                  columns = data.feature_names + ["target"])

# 8:2로 분할
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                    random_state = 42)
alpha = 1.0
lasso = Lasso(alpha = alpha)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"(MSE): {mse:.2f}") # 변화량을 봐야함 , 절대값으로 보면 안됨

# !pip install ace_tools_open

coefficients = pd.DataFrame({"Feature": [f"X{i}" for i in range(len(lasso.coef_))], "Coefficient": lasso.coef_})
import ace_tools_open as tools
tools.display_dataframe_to_user(name="회귀계수", dataframe = coefficients)
# 변수 선택법의 효과가 있음 (lasso 모델 자체가)

# 문제: 위에 변수를 실제 변수이름으로 대체하시오.

coefficients = pd.DataFrame({
    "Feature": [f"{data.feature_names[i]}" for i in range(len(lasso.coef_))], "Coefficient": lasso.coef_})
tools.display_dataframe_to_user(name="회귀계수", dataframe=coefficients)

# 비선형회귀
x = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
y = np.array([0.0,0.8,0.9,0.1,-0.8,-1.0])
z = np.polyfit(x,y,3)
z

# scikits에서는 비선형 회귀를 PolynomialFeatures로 구현
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3,2)
print(X)
# fit -> transform -> predict
poly = PolynomialFeatures(2)
poly.fit_transform(X) # 학습후 변환

# Ridge 모델
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
def f(x):
  return x * np.sin(x)

x_plot = np.linspace(0, 10, 100)
x = np.linspace(0,10,100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20]) # 정렬
y = f(x)

X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]
colors = ["teal", "yellowgreen", "gold"]
lw = 2
plt.plot(x_plot, f(x_plot), color="red", linewidth = lw, label = "ground truth")
plt.scatter(x,y,color="navy", s=30, marker="o", label = "training points")
# 비선형 회귀
# degree-2 polynomial features are [ 1, a, b, a^2, ab, b^2]
for count, degree in enumerate([3,4,5]):
  # 파이프 안에서 순차적으로 실행
  # make_pipeline : Pipeline 객체를 리턴 # 파이프 안에서 순차적으로 실행
  # make_pipeline : Pipeline 객체를 리턴
  # R %>%
  model = make_pipeline(PolynomialFeatures(degree),Ridge())
  # model.fit, predict, score, transform.fit, transform
  model.fit(X,y)
  y_plot = model.predict(X_plot)
  plt.plot(x_plot, y_plot, color = colors[count], linewidth=lw, label = "degree %d" % degree)
plt.legend(loc="lower left")
plt.show()

# 값 확인해보기
# Ridge는 예측 모델
# transformer (전처리) : 결측치, 이상치, 범주화, 정규화
# 데이터 차원변경, PCA 차원축소
model = make_pipeline(PolynomialFeatures(5), Ridge())
model.fit(X,y)
y_plot = model.predict(X_plot)
y_plot

# 자동 명명식
ridge_model = model.named_steps["ridge"]
print("계수 (coef_):", ridge_model.coef_)
print("절편 (intercept_):", ridge_model.intercept_)


# 문제 : MSE값과 r2_score값을 출력하시오
print("MSE: %.2f" % mean_squared_error(f(x_plot), y_plot))
print("결정계수: %.2f" % r2_score(f(x_plot), y_plot))

pipeline = model
print("단계:", pipeline.steps)

print("Ridge 계수:", pipeline.named_steps["ridge"].coef_)

print("모든 파라미터:", pipeline.get_params().keys())
# 외부에서 접근 ridge_alpha -> 매개변수의 이름도 자동으로 바뀜
pipeline.set_params(ridge__alpha=10)
pipeline.fit(X,y)

# pipeline + GridsearchCV는 함께 사용
# model_selection 4가지: train_test_split, parameter tunning,
# Cross Validation(CV), RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    "polynomialfeatures__degree": [2,3,4],
    "ridge__alpha": [0.1,1,10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5) # 9 * 5 = 45
grid.fit(X,y)
print("최적 파라미터:", grid.best_params_) # 최적의 파라미터 조합을 출력

print("ridge__alpha =", pipeline.get_params()["ridge__alpha"])
print("polynomialfeatures__degree =", pipeline.get_params()["polynomialfeatures__degree"])

for k, v in pipeline.get_params().items():
  print(f"{k}:{v}")

import pandas as pd
params = pipeline.get_params()
df = pd.DataFrame(params.items(), columns = ["Parameter", "Value"])
df

X = [[0,0], [1,1], [2,2]]
y = [0,1,2]
# 위의 데이터를 이용해서 LinearRegression을 실행하시오.
# 기울기와 절편을 확인해보시오
# 수식을 생성해 보시오
# 리턴 값 2차원 -> 1차원

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print("기울기(coef_):", model.coef_)
print("절편(intercept_):", model.intercept_)

X = np.array([[0,0], [1,1], [2,2]])
y = np.array([0,1,2])
reg = linear_model.LinearRegression()
reg.fit(X,y)
reg.coef_ # array([[0.5, 0.5]])

x1 = 0
x2 = 0
y = reg.coef_[0] * x1 + reg.coef_[1] * x2 + reg.intercept_
y

# 캘리포니아 집값 예측 데이터
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd
california = fetch_california_housing() # sklearn.utils import Bunch
type(california) # data, target, feature_name 구성

dfX = pd.DataFrame(california.data, columns = california.feature_names)
dfy = pd.DataFrame(california.target, columns = ["price"])
print(dfX.head())
dfy.head()

dfX.shape # (20640, 8)

model_california = LinearRegression().fit(
    california.data, california.target) # ndarray

model_california = LinearRegression().fit(dfX, dfy) # dataframe
predicitions = model_california.predict(california.data)
print(mean_squared_error(dfy, predicitions))
print(model_california.score(california.data, dfy))

# 정규화 : 수식 (5점문제) - 데이터 분석
# scale = (X- Xbar) / sd
# robust_scale = (X-median) / (quantiel(0.75) - quantiel(0.25))
# minmax_scale = (X-min) / (max-min)
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
print(np.arange(10, dtype = np.float64) - 3)
x = (np.arange(1000, dtype = np.float64) - 500).reshape(-1, 1)
print(x.shape)
# horizontal 수평으로 (열로 합쳐라) -> hstack
df = pd.DataFrame(np.hstack([x, scale(x), robust_scale(x), minmax_scale(x), maxabs_scale(x)]), columns = ["x", "scale(x)", "robust_scale(x)", "minmax_scale(x)", "maxabs_scale(x)"])
print(df.head())

# 문제 : 이 데이터를 수식으로 scale을 구현하시오
x = (np.arange(1000, dtype = np.float64) - 500).reshape(-1,1)

# 1) scale
# 2) robust
# 3) minmax를 수식으로 구현하시오



q1 = np.percentile(x, 25)
q3 = np.percentile(x, 75)
IQR = q3 -q1

# StandardScaler

x_scale = (x - np.mean(x)) / np.std(x)

# RobustScaler
q1 = np.percentile(x, 25)
q3 = np.percentile(x, 75)
IQR = q3 -q1

x_robust = (x - np.median(x)) / IQR

# MinMaxScaler
x_minmax = (x - np.min(x)) / (np.max(x) - np.min(x))

# maxabs_scale -> 절대값이 가장 큰것으로 나눔
x_maxabs = (x / np.max(np.abs(x)))

# 시각화
from sklearn.datasets import load_iris
iris = load_iris()
print(type(iris))
data1 = iris.data
print(data1.shape)
data2 = scale(iris.data)

print("전처리 전 평균:", np.mean(data1, axis=0))
print("전처리 후 std:", np.mean(data1, axis=0))
print("전처리 전 평균:", np.mean(data2, axis=0))
print("전처리 후 std:", np.mean(data2, axis=0))

sns.jointplot(data1[:,0]) # 분포 동일 ,형태 동일, 사이지만 변동
plt.show()
sns.jointplot(data2[:,0])
plt.show() # 상대적인 거리, 분포의 모양, 밀도비율은 불변

# 함수형, 클래스형
# scale(함수), StandardScaler(평균,표준편차를 기억)
# pipeline, gridsearchcv에서는 StandarScaler를 사용

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data1) # 학습 data1의 평균, 표준편차를 갖고 있음
data2 = scaler.transform(data1)
print(data1.std(), data2.std()) # 정규화된 것은 1.0

data3 = scaler.fit_transform(data1)
data3.std()



# data2를 복원하시오.
data2 * data1.std() + data1.mean()

# 원래의 사이즈로 원복
scaler.inverse_transform(data2)

from sklearn.preprocessing import MinMaxScaler
data1 = iris.data
minmax = MinMaxScaler()
res = minmax.fit_transform(data1)
res[0,:]

(data1 - data1.min()) / (data1.max()-data1.min())

# 원래의 값으로 복원
# 열기준
res * (data1.max() - data1.min()) + data1.min()

# L2 피타고라스, L1 절대값
# 행기준 or 열기준
# -> vector는 하나의 데이터를 기준 행단위로 normalize


from sklearn import preprocessing
X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
X_normalized = preprocessing.normalize(X, norm = "l2")
X_normalized

X = np.array(X)
norm = np.sqrt(np.sum(X[0, :] **2)) # sqrt(x**2 + y**2 + z**2)
print(X[0,:])
X[0,:] / norm

X_normalized = preprocessing.normalize(X, norm="l2")
X_normalized

# 행으로 정규화 : inver_transform이 없음
normalizer = preprocessing.Normalizer().fit(X)
normalizer.transform(X)

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values = np.nan, strategy = "mean")
# 채울 때는 열기준
imp_mean.fit([[7,2,3], [4, np.nan, 6], [10,5,9]])
X = [[np.nan,2,3], [4,np.nan,6], [10, np.nan, 9]]
print(X)
print(imp_mean.transform(X))

from sklearn.impute import KNNImputer
X = [[1,2,np.nan], [3,4,5], [np.nan,6,5],[8,8,7]]

imputer = KNNImputer(n_neighbors = 2) # 이웃하는 것 2개를 보고 채
imputer.fit_transform(X)

# 두 벡터의 거리값
np.sqrt((4-6)**2 + (3-5)**2)
np.sqrt((8-6)**2 + (7-5)**2)

from scipy.spatial import distance
a = np.array([1,0,0,0,1,1,1])
b = np.array([0,0,1,0,0,1,0])

print(distance.euclidean(a,b)) #
# knn(지도학습)-가까운 이웃(k개 보고 예측), kmeans(비지도)- 비슷한 데이터끼리 군집화
np.sqrt(np.sum((a-b)**2))


dist = np.linalg.norm(a-b) # 대각선 길이
a-b # 원점

# 직선거리로 구함
print("Manhattan:", distance.cityblock(a,b))

# 두 벡터의 코사인 유사도
# 밑변/ 빗변 -> 코사인 세타
# 벡터의 코사인 유사도 -> 두 벡터를 내적을 내고 크기값으로 나눈 값
print("cosine", distance.cosine(a,b))

# 이 코사인 유사도를 수식으로 나타내시오.
a = np.array([1,0,0,0,1,1,1])
b = np.array([0,0,1,0,0,1,0])

# 내적_요소끼리 곱해서 더함
np.sum(a * b) # 1
dotvalue = np.sum(a*b)

# 벡터의 크기(norm)
dotvalue / np.sqrt(np.sum((a**2))) * np.sqrt(np.sum((b**2)))

# 코사인 유사도_ 요소곱을 크기값으로 나눔
dotvalue / np.linalg.norm(a) * np.linalg.norm(b)

title = ["CRIM", "ZN", "INDUS", "CHAS", "NOX","RM", "AGE", "RAD",
         "TAX", "PTRATIO", "B", "LSTAT", "medv"]
data_url = "http://lib.stat.cmu.edu/datasets/boston" # 원본
raw_df = pd.read_csv(data_url, sep ="\s+", skiprows =22, header=None)
# 1개의 데이터가 2행으로 구성
# ::의 의미 : 2의 배수로 건너뛰면서
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,:2]])
target = raw_df.values[1::2,2]
print(type(data))
boston_df = pd.DataFrame(data, columns = title)
boston_df.head()

# 위 데이터를 이상치 처리
from scipy import stats
z = np.abs(stats.zscore(boston_df))
print(z.shape)
threshold = 3
print(np.where(z>3))

boston_df = boston_df[(z<3).all(axis=1)] # 표준점수 이용
boston_df.shape # (415,13)

# IQR을 통해서 이상치를 제거하시오.
# 열별로 IQR을 구해서 열마다 처리
# quantile은 기본이 열로 적용됨

boston_df.iloc[:,0].quantile(0.25)
boston_df.iloc[:,1].quantile(0.25)
# 데이터 프렘임이 열기반
Q1 = boston_df.quantile(0.25)
Q1

Q3 = boston_df.quantile(0.75)
Q3

IQR = Q3 - Q1
IQR
print(boston_df < (Q1 - 1.5 * IQR)) # 하한선
print(boston_df > (Q3 + 1.5 * IQR)) # 상한선

# (이 정도 식이 머리에서 나오면 코딩 안료)
boston_df_out = boston_df[~((boston_df < (Q1 - 1.5 * IQR)) | (boston_df > (Q3 + 1.5 * IQR))).any(axis=1)]
boston_df_out.shape

# OneHotEncode : 범주형 데이터를 숫자로 바꾸는 방법
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder() # 인스턴스
X = np.array([[2],[3],[4],[2],[4]]) # 5by1
ohe.fit(X) # 범주를 구함 2,3,4
ohe.categories_ # 15개의 기억공간 ,500만개(메모리 낭비) -> 0은 저장하지 않고 1인 것만 저장 => 희소행렬(sparce matrix)

print(ohe.transform(X)) # 희소행렬로 저장되어 있어 안을 볼 수 없음

# 사람이 볼 수 있는 정상 matrix로 바꿔주는 방법
print(ohe.transform(X).toarray())

# 파이썬 core에서 데이터가 객체에 담기면(list)로 봄
# zip -> 그냥 못 보고 -> list(zip) 이래야 보임
# 머신러닝에서는 toarray() : 거의 이거 사용

ohe.inverse_transform([[0.,0.,1.]])

ohe.fit([["서울"], ["서울"], ["대전"], ["부산"]])
# 서울, 대전 부산 -> 문자열은 범주가 자동 정렬됨
ohe.transform((["서울"],["서울"])).toarray()

# 범주가 범주인 경우
ohe = OneHotEncoder()
X = np.array([[0,0,4], [1,1,0],[0,2,1],[1,0,2],[1,1,3]])
ohe.fit(X)
ohe.categories_

ohe.transform(X).toarray()

ohe.inverse_transform([[1.,0.,1.,0.,0.,0.,0.,0.,0.,1.]])

# 첫번째열 범주 2개 + 두번째열 범주 3개 -> 총 5개로 표현
X = [["Male",1], ["Female", 2], ["Female", 3]]
onehot = OneHotEncoder(handle_unknown = "ignore") # 범주에 없는 것이 입력되면 0(무시)처리
onehot.fit(X)
print(onehot.categories_)
onehot.transform(X).toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit([1,2,2,6,4]) # 범주 4
le.classes_

le.transform([1,1,2,6]) # 범주화해라

le.inverse_transform([0,0,1,3])

# 딕셔너리 형태의 문서를 숫자 벡터로 변환해서 DTM(Document-Term Matrix)을 만드는 예제
# text mining: 문서 특징 추출
# AI는 의미 확률적으로 다음에 오는 단어
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
#          문서1                문서2
D = [{"foo":1, "bar":2}, {"foo":3, "baz":1}]
X = v.fit_transform(D)
# Document Trem matrix(DTM)
# 단어를 정렬
X

v.feature_names_