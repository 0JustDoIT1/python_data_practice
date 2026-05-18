# !lsb_release -a

import pandas as pd
print(pd.__version__)

print(pd.show_versions())
# pyarrow : backend : 메모리 표준 포맷 -> 다양한 변환

import numpy as np
# 행과 열 중심 가능
# 계산방향이 필요 : axis -> 없음 : 전체, 0 : 행방향 , 1 : 열방향
a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print("범위", np.ptp(a))
print("열범위", np.ptp(a, axis = 1)) # 3
print("열방향 분위수", np.percentile(a, 50, axis = 1))
print("분위수", np.percentile(a, 50)) # 전체
print("1사분위수", np.percentile(a, 25))
print("3사분위수", np.percentile(a, 75))
print("행중위수", np.median(a, axis = 0)) # 열결과
print("열평균", np.mean(a, axis = 1)) # 행결과
print("행분산", np.var(a, axis = 0))
print("행표준편차", np.std(a, axis = 0))
print("전체분산", np.var(a))
print("전체표준편차", np.std(a))

# 시리즈는 열한개 (1차원만 가능)
# axis가 없음 -> 기본이 열중심
a = np.array([1,2,3,4,6,8,9])
numbers = pd.Series(a)
print("합계", numbers.sum())
print("행중위수", numbers.median())
print("열평균", numbers.mean())
print("분산", numbers.var())
print("표준편차", numbers.std())
print("전체표준편차", numbers.std())
# 추가된 함수들
print("정상데이터", numbers.count()) # null을 제외하고 개수
print("기본통계설명", numbers.describe())
print("정상데이터", numbers.product())

# 문제
# 평균을 수식으로 구하시오
# 분산을 수식으로 구하시오
# 표준편차를 수식으로 구하시오
a_mean = np.sum(a) / len(a)
a_var = np.sum((a - a_mean)**2 / len(a))
a_std = np.sqrt(a_var)

print(a_mean)
print(a_var)
print(a_std)

print("방법1 (max-min):", a.max() - a.min())
print("방법2 (np.ptp):", np.ptp(a))
print("방법3 (agg):", numbers.agg(lambda x: x.max() - x.min())) # 열로 입력

courses = pd.Series(["numpy", "pandas", "scipy", "statsmodels", "scikits", "tensorflow", "keras", "pytorch"])
print(courses)
# 열이름을 주지 않으면 기본이 RangeIndex : 행 인덱스
print(courses.index) # dict형 함수
print("키")
print(courses.keys)
print("값")
print(courses.values)
print("타입")
print(courses.dtype) # numpy의 속성 정보 : object -> Pandas에서는 문자열은 object
print(courses.shape)
print(courses.ndim)
print(courses.size)
# series에 추가된 함수들
print("배열", courses.array) # 배열
print("헤더정보")
print(courses.head()) # 6개
print(courses.tail()) # 뒤에서
print(courses.describe())

courses.info()
# null을 중시

ser = pd.Series([20,25,15,10], ['Java', 'Spark', 'PySpark', 'Pandas'])
print(ser)
ser.index # Index

# 인덱스를 기준으로 입력
d = {'a': 1, 'b': 2, 'c': 3}
ser = pd.Series(data=d, index=['a', 'y', 'z'])
ser

# R하고 달라서 NaN도 데이터 처리함 (에러가 나지 않음)
def square(x):
  return x**3
ser2 = ser.apply(square)
ser2

ser['a']

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data

data['b']

data[1] # RangeIndex가 디폴트 생성

data.keys()

data.values

list(data.items())

data['e'] = 1.25 # 추가 가능
data

data['a':'c'] # 뒤에 있는 데이터 제외
data[0:2]

data[['a', 'e']]

data['a'] = 100 # 수정
data

area = pd.Series({'California': 423967, 'Texas': 695662, 'New York': 141297, 'Florida': 170312, 'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193, 'New York': 19651127, 'Florida': 19552860, 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data

data['area'] # 열 : 행렬구분이 어려움
data.area

# 숫자 인덱싱하고 문자 인덱싱을 구분하자
print(data.loc['California'])

print(data.loc[['Texas', 'Florida']]) # 행2개

print(data.loc['Texas', 'pop'])
print(data.loc[['Texas', 'Florida'], ['pop']])

print(data.iloc[0])

print(data.iloc[0, 1])

data['density'] = data['pop'] / data['area']
data

data.values

data.T # 전치행렬 : 이질적데이터간도 가능

data.iloc[:3, :2]

# 숫자는 마지막이 제외, 문자는 포함
data.loc[:"Illinois", :'pop']

# density>100 큰 'pop', 'density' 열을 출력하시오
# density의 California의 데이터를 90으로 수정하시오
data.loc[data.density > 100, ['pop', 'density']]
data.loc['California', 'density'] = 90
data

# data.iloc[data.density > 100] 숫자일 때는 조건식이 안됨
# boolean indexing

mask = [False, True, True, False, False]
print(data.iloc[mask])

print(data.sort_values('pop', ascending=False))

# 인덱스는 순서가 보장
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(s)
s.reindex(['e', 'b', 'f', 'd'])

data = np.array(['a', 'b', 'c', 'd', 'a'])
s = pd.Series(data)
print("시리즈 인덱스", s.index)
print("시리즈 인덱스의 값 ", s.index.values)
print(s.value_counts())
print("상대도수분포표", s.value_counts(normalize=True))
print(s.describe())
print("데이터 타입은 : ", s.dtypes) # s가 붙음 (열별로)
print("차원", s.ndim)
print("차수", s.shape)

# 데이터 프레임연산

# 연산은 가능한데 결과가 NaN
sdata = { 'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000} # California가 없음
obj3 = pd.Series(sdata)
print(obj3)
states = ['California', 'Ohio', 'Oregon', 'Texas'] # Utah 없음
obj4 = pd.Series(sdata, index=states)
print(obj4)
print("null이 있는가", pd.isnull(obj4))
print("결측치가 아닌 데이터", pd.notnull(obj4))
print("연산결과를 출력.", obj3 + obj4)

# obj4의 index 값을 출력하시오
# 벡터화된 문자열 함수
pd.Series(obj4.index.values).str.upper()

pd.Series(obj4.index.values).str.count('a')

result = obj3 + obj4
df = pd.DataFrame({
    "obj3": obj3,
    "obj4": obj4,
    "sum": result
})
print("DataFrame")
print(df)
print()

parquet_file = "series_data.parquet"
df.to_parquet(
    parquet_file,
    engine="pyarrow",
    index=True
)

loaded_df = pd.read_parquet(
    parquet_file,
    engine="pyarrow"
)
print(loaded_df)

# Bunch
from sklearn.datasets import load_iris
iris = load_iris()
type(iris)

iris['data']
iris['target'] # 종속변수
iris['feature_names']

# column
iris = pd.DataFrame(data = np.c_[iris['data'], iris['target']], columns = iris['feature_names'] + ['target'])

iris.info()

iris.head()

iris.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
iris['Name'] = iris['Name'].astype('int32')
iris.head()

iris.plot()

import seaborn as sns
sns.pairplot(iris.iloc[:, :4], height=2) # 종속변수 제외
# 상관도

iris["sepal_ratio"] = iris['SepalWidth'] / iris['SepalLength']
iris.head()

a = (iris.query('SepalLength > 5')).assign(
    # 파생변수 -> 메모리에 남지 않음 (중간 작업 결과는 메모리 해제)
    SepalRatio = lambda x: x.SepalWidth / x.SepalLength,
    PetalRatio = lambda x: x.PetalWidth / x.PetalLength
).plot(kind='scatter', x='SepalRatio', y='PetalRatio')

a = (iris.query('SepalLength > 5')
  .assign(
      SepalRatio = lambda x: x.SepalWidth / x.SepalLength,
  )
  .groupby("Name")
  .agg(
      Count=("Name", "count"),
      AvgSepalLength=("SepalLength", "mean"))
  .sort_values(
      "AvgSepalLength",
      ascending=False
  )
)
print("분석 결과")
print(result)

#  'Name':pd.Series(['김하나','이하나','삼하나','사하나','오하나', '육하나','칠하나', '팔하나'])
#  'Age':pd.Series([25,26,25,23,30,29,23])
#  'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])

# 문제 : 위 데이터를 데이터 프레임으로 작성하시오
# 문제 1) 데이터 타입을 확인하시오
# 문제 2) 차원을 확인하고 차수를 확인하시오
# 문제 3) 전체 데이터 요소 사이즈를 확인하시오
# 문제 4) 값들만 출력하시오
name = pd.Series(['김하나','이하나','삼하나','사하나','오하나', '육하나','칠하나', '팔하나'])
age = pd.Series([25,26,25,23,30,29,23])
rating = pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])
df = pd.DataFrame({'name': name, 'age': age, 'rating': rating})
df

df.dtypes # 타입
df.ndim # 차원
df.shape # 차수
df.size # 사이즈
df.values # dict 속성

print('축', df.axes) # 행, 열 인덱스
print('행인덱스', df.index) # 행 인덱스
print('열인덱스', df.columns) # 열 인덱스
print('비었나', df.empty)
print(df.head(3))
print(df.tail(2))

df

print("합계", df.sum())
print("합계", df.sum(numeric_only=True))

print("상관계수", df.iloc[:, [1,2]].corr())

df.describe() # 숫자 변수로만 제한

print('오브젝트', df.describe(include=['object'])) # 일부 데이터타입
print(df.describe(include='all'))

# 문제
data={'state': ['경기', '강원', '서울', '충북', '인천'],
      'year': [2000, 2001, 2002, 2001, 2002],
      'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

frame2 = pd.DataFrame(data, index=['one', 'two', 'three', 'four', 'five'])
frame2

# 1) state만 출력하시오
frame2.state
# 2) debt 열을 추가하고 모든 값을 16.5로 입력하시오
frame2['debt'] = 16.5
frame2
# 3) 파생변수 estern에 주소가 서울인가를 따져 서울이면 True 아니면 False를 입력하시오
# test = frame2.assign(
#     estern = lambda x: x.state == '서울'
# )
# print(test)
frame2['estern'] = frame2['state'] == '서울'
frame2
# 4) debt 열에 대해서 (-1.2, -1.5, -1.7) index=['two', 'four','five']로 된 시리즈를 만들고 대체하시오
frame2['debt'] = pd.Series([-1.2, -1.5, -1.7], ['two', 'four', 'five'])
frame2
# 5) 모든 열의 데이터를 요약해서 출력하시오
frame2.describe(include="all")
# 6) state 열의 도수 분포표를 작성하시오 (열)
frame2.state.value_counts()
# 7) year를 중심하고 정렬하시오 (sort_values)
frame2.sort_values('year') # 값을 기준으로 정렬

print(frame2.state.sort_index()) # 인덱스를 기준으로 정렬

# time index
# datetime : 시간 : 문자, 숫자(계산) -> 입출력 문자
# 정밀성 : 시분초 년월일 : 초 : ms(1/1000), micros(1/1000000), ns(1/1000000000)
# numpy : nano second : datetime64
# DatetimeIndex 날짜를 인덱스 (한시점 : log발생, event, sensor)
# RangeIndex, Index
# Period Index : 분기별, 월별
# TimDeltaIndex: 한시점  - 다른 한시점을 뺀 것 = 5days
# IntervalIndex : (1 ~ 2) (2 ~ 3)

df = pd.DataFrame({"y": [1, 2, 3]},
                  index = pd.to_datetime(["2000-03-31 00:00:00",
                                          "2000-05-31 00:00:00",
                                          "2000-08-31 00:00:00"]))
print(df)
print(df.axes) # 행인덱스 : DatetimeIndex, 열인덱스 Index
df.index=df.index.to_period("M") # Month 월별
df.index=df.index.to_period("Q") # Quarer 분기별
df.index=df.index.to_period("Y") # Year 연별
df

# linux에서 1970.1.1 일 이후로 경과된 초
df_timedelta = df.copy()
base_date = pd.Timestamp("2000-01-01")
df_timedelta.index = df_timedelta.index - base_date
print(df_timedelta)

# RangeIndex
df_range = df.reset_index(drop=True)
print(df_range)

import matplotlib.pyplot as plt
data = np.random.randn(25, 4) # 표준 정규분포
df = pd.DataFrame(data, columns=list('ABCD'))
ax = df.plot.box()
plt.show()

# 문제
data = {"서울" : ["수도권", 9904312, 9631482, 9762546, 9853972, 0.0283],
        "부산" : ["경상권", 3448737, 3393191, 3512547, 3655437, 0.0163],
        "인천" : ["수도권", 2890451, 2632035, 2517680, 2466338, 0.0982],
        "대구" : ["경상권", 2466052, 2431774, 2456016, 2473990, 0.0141]}
columns = ['지역','2018', '2017', '2016', '2015', '2015-2018증가율']
# 위 데이터를 데이터프레임으로 생성하시오
df = pd.DataFrame(data=data, index=columns)
df = df.T
df

df.info()
# 1) 2015-2018년 증가율을 %로 변경하시오
df["2015"] = df["2015"].astype("float")
df["2016"] = df["2016"].astype("float")
df["2017"] = df["2017"].astype("float")
df["2018"] = df["2018"].astype("float")
df["2015-2018증가율"].astype("float")

df["2015-2018증가율"] = df["2015-2018증가율"] * 100
# 2) 2015-2017년 증가율을 구해서 변수로 추가하시오
#   (2015-2018 증가율)
df["2015-2017증가율"] = ((df["2017"] - df["2015"]) / df["2015"] * 100). round(2)
df

# 구간범주화
df["growth_level"] = pd.cut(
    df["2015-2017증가율"],
    bins=[-10, 0, 5, 10],
    labels=["낮음", "중간", "높음"]
)
print(df.loc["서울", "growth_level"])

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

plt.rc('axes', unicode_minus=False)

# 한글처리
df.plot(
    x="지역",
    y="2015-2017증가율",
    kind="bar"
)

index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]

pop = pd.Series(populations, index=index)
pop

pop[('California', 2010):('Texas', 2000)]

pop.index

pop[[i for i in pop.index if i[1] == 2010]]

index = pd.MultiIndex.from_tuples(index)
index

pop = pop.reindex(index)
pop

pop[:, 2010]

# 행을 열로 전환
pop_df = pop.unstack()
pop_df

# 프레임이 프레임 안에 위치할 때 행인덱스가 유지
pop_df = pd.DataFrame({'total': pop,
                       'under18': [1,2,3,4,5,6]})
pop_df

# 학생 3명을 입력하시오
# 이름, 국어, 영어, 수학 점수 입력
students = []
name = []
for i in range(3):
  student = input("학생의 이름, 국어, 영어, 수학 점수를 입력하시오")
  data=student.split(",")
  students.append(data[1:])
  name.append(data[0])

data = pd.DataFrame(students, index=name, columns=["국어", "영어", "수학"])
data

data.info()

# astype : raise '에러발생', 'coerce': NaN
data['국어'] = pd.to_numeric(data['국어'])
data['영어'] = pd.to_numeric(data['영어'])
data['수학'] = pd.to_numeric(data['수학'])
data.info()

data.describe()

# 국어, 영어, 수학 합계
data.iloc[:, :].sum()
data.iloc[:, :].mean()

data.apply(lambda x: x.mean(), axis=0).round(2)

data["합계"] = data.iloc[:,:3].apply(lambda x: x.sum(), axis = 1)
data

data["평균"] = data.iloc[:,:3].apply(lambda x: x.mean(), axis = 1).round(1)
data

# 합계를 중심으로 정렬하시오
data = data.sort_values("합계", axis=0, ascending=False)
# data['등수'] = data.index + 1
data['등수'] = range(1, len(data) + 1)
data



# 과목별 분산을 출력하시오
data.iloc[:, :3].apply(lambda x: x.var(), axis=0).round(2)

data.iloc[:,:].cov() # 공분산 행렬

data.iloc[:,:3].corr()

print(data.iloc[:, :3].apply(['var', 'std']))

plt.rcParams.update({'font.size': 16, 'font.family': 'NanumGothic'})

sns.pairplot(data.iloc[:,:3])

plt.matshow(data.iloc[:,:3].corr())
plt.colorbar() # 컬러에 따라 값 매핑 바
plt.show()

# !pip install pandasql

from pandasql import sqldf

# 전역적 실행
run_query = lambda query: sqldf(query, globals())

import seaborn as sns
tips_df = sns.load_dataset("tips")
tips_df.head()

query_1 = """
SELECT *
FROM tips_df
LIMIT 10;
"""
result_1 = run_query(query_1)
print(result_1)

query_2 = """
SELECT *
FROM tips_df
WHERE total_bill > 30 AND tip > 5;
"""

result_2 = run_query(query_2)
print(result_2)

result_2 = tips_df[
    (tips_df['total_bill'] > 30) & (tips_df['tip'] > 5)
]
print(result_2)

# !pip install pymysql

import pymysql
conn = pymysql.connect(host="104.198.27.181", port=3306, user="hbc3869", passwd="Zmfjszl123!", db="daejeon", charset="utf8", autocommit=True)

query = f"CALL select_one({"김종호"});"
df = pd.read_sql_query(query, conn)
conn.close()

# !pip install duckdb # 메모리 DB - 고속 처리

import duckdb
conn = duckdb.connect(database="db.duckdb")
conn.sql("CREATE TABLE IF NOT EXISTS test_table (i INTEGER, j STRING)")
conn.sql("INSERT INTO test_table VALUES (1, 'one'), (9, 'nine')")
conn.table('test_table').show()

res = conn.sql(f""" SELECT * FROM test_table """)
res.fetchall()
print(res.df())

conn.close()

from google.colab import drive
drive.mount('/content/drive')

result = duckdb.sql("""
SELECT *
FROM '/content/drive/MyDrive/AI빅데이터_파이썬/dataset/iris.csv'
WHERE column0 > 5.0
""").df()
print(result.head())

conn = duckdb.connect()
conn.sql(
    f"""
    CREATE TABLE iris_table AS
    SELECT * FROM read_csv(
    '/content/drive/MyDrive/AI빅데이터_파이썬/dataset/iris.csv',
    AUTO_DETECT=TRUE);
    """
)

conn = duckdb.connect('/content/drive/MyDrive/AI빅데이터_파이썬/dataset/iris.csv')
conn.sql(
    f"""
    CREATE TABLE IF NOT EXISTS iris_table AS
    SELECT * FROM read_csv(
    '/content/drive/MyDrive/AI빅데이터_파이썬/dataset/iris.csv',
    AUTO_DETECT=TRUE);
    """
)
conn.sql("SELECT * FROM iris_table LIMIT 5")

# Bunch 데이터 -> 그 데이터를 duckdb에 저장
# 똑같은 작업을 duckdb 쿼리를 이용해서 데이터를 가지고 온 다음
# dataframe 작업으로 시각화하시오