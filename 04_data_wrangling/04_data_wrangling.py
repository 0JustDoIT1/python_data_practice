# series에만 적용되는 명령어들
import pandas as pd
df = pd.DataFrame({
    "student": ["Kim", "Lee", "Park", "Choi", "Jung", "Han", "Song", "Lim"],
    "gender": ["남", "여", "남", "여", "남", "여", "남", "여"],
    "city": ["Seoul", "Busan", "Seoul", "Daegu", "Busan", "Seoul", "Daegu", "Busan"],
    "score": [88, 95, 70, 82, 91, 65, 77, 89],
    "sales": [120000, 250000, 90000, 180000, 300000, 70000, 150000, 210000],
    "email": [
        "kim@gmail.com",
        "lee@naver.com",
        "park@gmail.com",
        "choi@daum.net",
        "jung@naver.com",
        "han@gmail.com",
        "song@daum.net",
        "lim@naver.com"
    ],
    "join_date": pd.to_datetime([
        "2026-01-05", "2026-01-10", "2026-02-03", "2026-02-20",
        "2026-03-01", "2026-03-15", "2026-04-01", "2026-04-10"
    ])
})

print(df)

# 범주화
# map : series에 함수를 적용할 때 사용
df["gender_code"] = df["gender"].map({ "남" : 1, "여" : 0})
print(df[["student", "gender", "gender_code"]])

# 상대 도수
print(df["city"].value_counts(normalize = True))

print(df["city"].unique()) # 결과는 'numpy.ndarray'타입
print(type(df["city"].unique()))

print(df["city"].nunique())
print(df["gender"].nunique())

# argmax, argmin 인덱스 (numpy)
# idxmax, idxmin (series)

# pandas 행인덱스, 열인덱스 접근
max_score_index = df["score"].idxmax()
print("최고 점수 인덱스:", max_score_index)
print(df.loc[max_score_index])

# nlargest vs nsmallest
print(df["score"].nlargest(3))
top3_index = df["score"].nlargest(3).index
print(df.loc[top3_index])

# 문제 : email 주소에서 도메인 주소만 추출하시오
# 벡터화된 문자열 함수
df["email_domain"] = df["email"].str.split('@').str[1]
print(df[["student", "email", "email_domain"]])
# gmail이 포함된 것만
gmail_users = df[df["email"].str.contains("gmail")]

df["join_date"]

df["year"] = df["join_date"].dt.year
df["month"] = df["join_date"].dt.month
df["day"] = df["join_date"].dt.day
df["weekday"] = df["join_date"].dt.day_name()

print(df[["student", "join_date", "year", "month", "day", "weekday"]])



# 월별 가입자수 도수를 출력하시오
# sort_index는 인덱스를 중심으로 정렬
# 방법 1
df["join_date"].dt.month.value_counts().sort_index()
# 방법 2
df["month"].value_counts().sort_index()

# merge
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
# pandas에서는 key 중복 가능
df1 = DataFrame({ 'key' : ['b','b','a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
print(df1)
df2 = DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
print(df2)
print(pd.merge(df1, df2, how = 'inner'))  # 결과에 c,d 없음
print(pd.merge(df1, df2, on = 'key')) # 기준컬럼
print(pd.merge(df1, df2, left_on = 'key', right_on = 'key'))
print(pd.merge(df1, df2, how = 'outer')) # key가 일치하지 않아도 모두 병합
# merge, join 후 형태확인, NaN 확인해야 함

# join은 index기반
data1 = { "name": ["대한", "민국", "만세"],
         "age": [50, 40, 30]}
data2 = { "qualified": [True, False, False]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
newdf = df1.join(df2)
newdf
data3 = { "name" : ["영원"], "age": [30]}
df3 = pd.DataFrame(data3)
# join이 안됨 - 열이름이 같아서
# newdf = df1.join(df3)
# print(newdf)
# 이름을 달리해서 join
newdf = df1.join(df3, lsuffix = "_left", rsuffix = "_right")
print(newdf)

# 다음을 id를 중심으로 merge 하시오
df1 = pd.DataFrame({"id": [1, 2, 3], "val1": ["A", "B", "C"]})
df2 = pd.DataFrame({"id": [2, 3, 4], "val2": ["X", "Y", "Z"]})
print(pd.merge(df1, df2, on = 'id'))
print(pd.merge(df1, df2, on = 'id', how = 'outer'))
print(pd.merge(df1, df2, on = 'id', how = 'inner'))

# index를 중심으로
df1_indexed = df1.set_index("id")
df2_indexed = df2.set_index("id")
print(df1_indexed.join(df2_indexed, how = "inner"))

print(pd.concat([df1, df2], axis = 1))

print(pd.concat([df1, df2], axis = 0))

# 데이터 중복 확인
# * 요소곱 : 반복
data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
print(data)
print("중복")

print(data.duplicated())
print(data.drop_duplicates())  # 중복(True) 제거
print(data.drop_duplicates(['k1']))
data.drop_duplicates(['k1', 'k2'], keep = 'last')

# 함수적용(map, apply, applymap)
frame = pd.DataFrame((np.random.randn(4, 3)),
                     columns = list(['rain', 'income', 'tax']),
                     index = ['seoul', 'daejun', 'incheon', 'daegu'])
print("absolute함수 적용\n", np.abs(frame))  # 절대값
print(frame.apply(np.ptp, axis = 0))  # ptp는 전체 데이터를 대상으로
# print("apply", frame.applymap(np.ptp, axis = 0)) 에러는 없는데 의미가 달라짐
# axis = 0 행방향 -> 결과는 열에 대해서 계산  , 열별 범위값
# axis = 1 행별 범위값
print(frame.apply(np.ptp, axis = 1))
f = lambda x: x.max() - x.min()
print("함수 객체의 행 적용 (열방향)\n", frame.apply(f, axis = 0))
print("함수 객체 열 적용 (행방향)\n", frame.apply(f, axis = 1))

# applymap
# apply에서는 열 통째로 함수로 전달
def f(x):
  return pd.Series([x.min(), x.max()], index = ['min', 'max'])
print("함수 객체 적용", frame.apply(f))
format = lambda x: '%2f' %x
print(frame.applymap(format)) # 요소가 함수로 입력

# map
# Series
print(frame['rain'].map(format)) # 요소가 함수로
print("결과")
print(frame['rain'].map(np.ptp)) # 열로 적용하는 함수(의미가 없음)
print(frame['rain'].apply(np.ptp)) # 열별적용

# 원래 속성 유지 - 데이터프레임
print(frame[['rain']].apply(np.ptp))
type(frame[['rain']]) # 데이터프레임
type(frame['rain']) # 시리즈

# 데이터프레임과 Series는 적용함수가 다르다

# 문제 : tax 필드에 100을 곱하고 10을 더하는 계산을 해서 파생변수를 생성하시오
ramv = lambda x: x * 100 + 10
def ftax(x):
  return x * 100 + 10
frame["tax_a"] = frame['tax'].map(ftax)
frame
frame["tax_a"] = frame['tax'].map(ramv)
frame

data = Series([1., -999., 2., -999., -1000., 3.])
print(data)
print("일정데이터를 nan으로 \n", data.replace(-999, np.nan))
print("변경", data.replace([-999, -1000], np.nan))
print("짝으로", data.replace([-999, -1000], [np.nan, 0]))
print(data.replace({-999:np.nan, -1000:0}))

# Series는 동질적 데이터만 입력 가능
data = pd.Series([1, np.nan, 'hello', None])
data

data.info()

data.isnull().sum() # True인 것만 합계

data[data.notnull()] # boolean indexing

data = data.dropna()
data

df = pd.DataFrame([[1,     np.nan,     2],
                   [2,       3,        5],
                   [np.nan,  4,        6]])
df

# 행을 중심하고 삭제
df.dropna()

df.dropna(axis = 'columns') # 열중심 삭제

df[3] = np.nan
print(df)
df.dropna(axis = 'columns', how = 'all')

df.fillna(0)

# backward fill 뒤의 값으로 채움
df.fillna(method = 'bfill', axis = 1) # 열방향으로
# ffill : forward fill

import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/dataset/pima.csv')
msno.matrix(df)
plt.show

df.iloc[0, 0] = np.nan
msno.bar(df)
plt.show()

# 클래스
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'median') # most_frequent, constant
imp.fit([[1, 2, 3], [np.nan, 3, 3], [7, 6, 3]]) # 학습
# [1, 2, 3]
# [np.nan, 3, 3]
# [7, 6, 3]
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp.transform(X)) # 변형 (nan을 채움)

# 유사함 : cos 유사도, 거리값
# K nearest neighbor : 피타고라스 거리값을 사용
from sklearn.impute import KNNImputer
X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
print(X)
# [1,      2,   np.nan],
# [3,      4,   3],
# [np.nan, 6,   5],
# [8,      8,   7]

imputer = KNNImputer(n_neighbors = 2)
imputer.fit_transform(X)



# numpy(corrcoef) -> pandas(corr) -> scipy(분석) -> statsmodels(모델)
from scipy import stats
import numpy as np
df = pd.DataFrame({'key': [1, 1, 1, 1, 100, 1, 1, 1, 1],
                   'data1': [1, 1, 1, 1, 1, 1, 1, 1, 20000]})

# z점수 기반 이상치 filtering
# zscore = (X = Xbar) / sd
z_scores = stats.zscore(df)
print(z_scores)
abs_z_scores = np.abs(z_scores) # 절대값
print(abs_z_scores)
# 2 : 표준편차 2배수
filtered_entries = (abs_z_scores < 2).all(axis = 1)
new_df = df[filtered_entries]
new_df

df = pd.DataFrame({'Data': np.random.normal(size = 200)})
print(df.head())
print(df[np.abs((df.Data - df.Data.mean()) / df.Data.std()) <= (3 *df.Data.std())])

# 분위수 기반 이상치 제거
df = pd.DataFrame({'Data': np.random.normal(size = 200)})
print(df.describe())
q_low = df.iloc[:, 0].quantile(0.01) # 1분위수  (하한값)
q_hi = df.iloc[:, 0].quantile(0.99)  # 99분위수 (상한값)
print("하한값", q_low)
print("상한값", q_hi)
# 상한값보다 작거나 하한값보다 크면 정상 데이터
df_filtered  = df[(df.iloc[:, 0] < q_hi) & (df.iloc[:, 0] > q_low)]
print("정상데이터", len(df_filtered))
df_filtered.head()

# 문제 : 비정상 데이터만 출력하시오
df[ ~((df.iloc[:, 0] < q_hi) & (df.iloc[:, 0] > q_low))]

# 문제: 사분위수 이상치 filtering을 참고해서 boxplot 이상치 filter를 구현하시오
IQR = df.iloc[:, 0].quantile(0.75) - df.iloc[:, 0].quantile(0.25)
q_hi = df.iloc[:, 0].quantile(0.75) + 1.5 * IQR
q_low = df.iloc[:, 0].quantile(0.25) - 1.5 * IQR
print("하한값", q_low)
print("상한값", q_hi)
df_filtered = df[(df.iloc[:, 0] < q_hi) & (df.iloc[:, 0] > q_low)]
print("정상 데이터", len(df_filtered))
df_filtered.head()

# 정규화 (z점수)
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
sample = df.head()
print("원본 데이터:\n", sample)
z_normalized = (sample - sample.mean()) / sample.std()
print("\nZ-점수 정규화된 데이터:\n", z_normalized)
z_normalized = (df - df.mean()) / df.std()
z_normalized

# 역정규화(z점수데이터 -> 원본 데이터로 변환)
# z = (x - mu)
#     ---------
#       std
# z * std = (x-mu)
# z * std + mu = x
original = (z_normalized * df.std()) + df.mean()
print("\n역표준화 결과")
print(original.head())

# Robust 정규화
median = df.median()
iqr = df.quantile(0.75) - df.quantile(0.25)
robust_scaled = (df - median) / iqr
print("\n중앙값:\n", median, "\nIQR:\n", iqr)
print("\nRobust 정규화된 데이터:\n", robust_scaled)
# 역정규화
original = (robust_scaled * iqr) + median
print("\n역표준화 결과")
print("original.head()")

from sklearn import preprocessing
df = pd.DataFrame({'Data': np.random.normal(size = 200)})
print(df.describe())
df['X_scale'] = preprocessing.scale(df)
df

df['X_log'] = preprocessing.scale(np.log(df["Data"] + 1))
df['X_sqrt'] = preprocessing.scale(np.sqrt(df["Data"] + 1))

from scipy import stats
def normality_test(series, name):
  stat, p = stats.shapiro(series)
  print(f"{name} : W = {stat:.3f}, p = {p:.3f} ->",
  "정규분포 가정 O" if p > 0.05 else "정규분포 가정 X")
  return p
pvals = {}
for col in ['X_scale', 'X_log', 'X_sqrt']:
  pvals[col] = normality_test(df[col], col)
best_transform = max(pvals, key = pvals.get)
print("\n✅ 최적 변환 컬럼:", best_transform)

# !pip install mglearn

import mglearn
import matplotlib.pyplot as plt
mglearn.plots.plot_scaling()

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)  # 평균, 표준편차 계산
print("평균:", scaler.mean_)
print("표준편차:", scaler.scale_)
X_scaled = scaler.transform(X_train) # 변환
print("\n스케일링된 데이터:\n", X_scaled)
X_inversed = scaler.inverse_transform(X_scaled)
print("\n복원된 원래 데이터:\n", X_inversed) # 예측값
# 예측값 != 실제값

min_max_scaler = preprocessing.MinMaxScaler()
# min, max를 계산
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax # 정규화된 값

X_train_minmax * min_max_scaler.data_range_ + min_max_scaler.data_min_

# 범주화(Categorical 데이터 타입을 생성)
# 균등 구간
df = pd.DataFrame({'value': np.random.randint(0, 100, 20)})
print(df.shape)

labels = [ "{0} - {1}".format(i, i + 9)
  for i in range(0, 100, 10)]
print(labels)

df['group'] = pd.qcut(df.value, 10, labels = labels)
print(df.head(20))
quantiles = pd.qcut(df.value, 10, labels = labels)
type(quantiles) # pandas.core.series.Series
quantiles

list(range(0, 101, 10))

# 구간값을 부여하는 cut
# 비균등 구간값도 가능
labels = [f"{i} - {i + 9}" for i in range(0, 100, 10)]
df['group'] = pd.cut(df.value, bins = range(0, 101, 10), labels = labels, right = False) # 구간값 중에 왼쪽 포함
df.head(20)

import seaborn as sns
sns.displot(df, x = 'value', hue = 'group', element = 'step', bins = 10)

s = pd.Series(["a", "b", "c", "a"], dtype = "category")
s

df = pd.DataFrame({"A": ["a", "b", "c", "a"],
                   "C": [10, 20, 30, 40]})
df["B"] = df["A"].astype('category')
df.info()

# 범주형 시각화 ( seaborn 범주형 )
sns.catplot(x = 'B', y = 'C', data = df)
plt.show()

values = np.array([1, 2, 1, np.nan])
codes, uniques = pd.factorize(values)
print(codes)
print(uniques) # 범주값

# one-hot-encoding
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b', 'c', 'd'],
                   'data1': range(8)})
print(df)
print(pd.get_dummies(df['key'])) # 범주 -> 변수
dummies = pd.get_dummies(df['key'], prefix = 'key') # 접두사
print(dummies)
df_with_dummy = df[['data1']].join(dummies) # 행으로 join
print(df_with_dummy)

# 문제
import numpy as np
path = '/content/drive/MyDrive/dataset/olive.csv'
df = pd.read_csv(path)
print(df.shape)
print(df.head(5))
print(df.dtypes)
print(df.describe())
# 올리브 오일 구성성분으로 분류
# 포화 지방산 이름들 + 불포화 지방산  -> 재배 지역 분류 ( 품질 평가 )


# 1. 첫번째 컬럼의 이름을 ID_area로 지정하시오
df = df.rename(columns = {'Unnamed: 0':'ID_area'})
df.head()


# 2. region이 몇개의 범주로 이뤄졌는지 확인하시오
print(df["region"].nunique())

# 3. area도 같은 처리를 하시오
print(df["area"].nunique())

# 4. 처음 컬럼(Unnamed) 에 들어온 지역명 앞 숫자를 제거하시오
df["ID_area"] = df["ID_area"].astype(str).str.split('.').str[1]
print(df["ID_area"])
df.head()

print(df["ID_area"].astype(str).str.split('.').head())

df.columns

column = np.array(df.columns)
column[0] = 'ID_area'
df.columns = column
df

# 방법 2
df.rename(columns = {'Unnamed:0': 'ID_area'}, inplace = True)
print(df.columns)
df.head


df['region'].unique

print('area\t', df.area.unique())

df["ID_area"].unique()[:5]

df["ID_area"] = df["ID_area"].apply(lambda x: x.split('.')[1])
print(df.head())

acidlist = ['palmitic', 'palmitoleic', 'stearic', 'oleic']
dfsub = df[acidlist]
dfsub.head()

dfsub = df[acidlist].applymap(lambda x: x / 100.0)
dfsub.head()

df[acidlist] = dfsub
df.head()

# 과제 : 전처리 최적화
# 데이터타입 다운캐스팅, 범주화, parquet로 저장하고 로딩(이득)
from sklearn.datasets import load_iris
iris = load_iris(as_frame = True)
df = iris.frame
df.columns = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "Target"
]
df.head()
df

# 문제 1 : 0: "setosa", 1: "versicolor", 2: "virginica"로 target을 설명하는 Species 변수를 추가하시오
df["Species"] = df["Target"].map({0 :"setosa" , 1: "versicolor", 2: "virginica"})
df

# 문제 2 : 데이터의 메모리 사용량을 확인하시오
df.info()
# emory usage: 7.2+ KB


# 문제 3 : 데이터 타입을 Downcasting(float32) 하시오 (14M -> 3KB)
df.iloc[:, :5] = df.iloc[:, :5].astype('float32')
df.info()
# memory usage: 6.6+ KB

# 문제 4: Species 변수를 범주화하시오
df["Species"] = df["Species"].astype('category')
print(df.dtypes)
print(df.info())

# 문제 5: parquet 포맷으로 데이터를 저장하고 로딩하시오(파일 저장)
parquet_file = "iris_data.parquet"
df.to_parquet(parquet_file, engine = "pyarrow", index = True)


# 과제 : 전처리 최적화
# 데이터타입 다운캐스팅, 범주화, parquet로 저장하고 로딩(이득)
from sklearn.datasets import load_iris
iris = load_iris(as_frame = True)
df = iris.frame
df.columns = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "Target"
]
df.head()
df


# 문제 1 : 0: "setosa", 1: "versicolor", 2: "virginica"로 target을 설명하는 Species 변수를 추가하시오
df["Species"] = df["Target"].map({0 :"setosa" , 1: "versicolor", 2: "virginica"})
df

# 문제 2 : 데이터의 메모리 사용량을 확인하시오
print(df.info())
# emory usage: 7.2+ KB


# 문제 3 : 데이터 타입을 Downcasting(float32) 하시오 (14M -> 3KB)
df[df.columns[:5]] = df[df.columns[:5]].astype('float32')
# 데이터 타입 확인
print(df.dtypes)
# 메모리 사용량 확인
print(df.info())


# 문제 4: Species 변수를 범주화하시오
df["Species"] = df["Species"].astype('category')
print(df.dtypes)
print(df.info())

# 문제 5: parquet 포맷으로 데이터를 저장하고 로딩하시오(파일 저장)
parquet_file = "iris_data.parquet"
df.to_parquet(parquet_file, engine = "pyarrow", index = True)


loaded_df = pd.read_parquet(parquet_file, engine = "pyarrow")
print(loaded_df)