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

# matplotlib 기반 seaborn
import seaborn as sns
sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "NanumGothic",
        "axes.unicode_minus": False,
    }
)

import pandas as pd
import numpy as np
df = pd.DataFrame({
    "범주" : ["A", "B", "C", "A", "B"]
})
# df = pd.factorize(df["범주"])
print(df) # array
print(type(df))

codes, uniques = pd.factorize(df["범주"])
df["범주코드"] = codes # 범주화

df["범주"] = pd.Categorical(df["범주"])
print(df)

# 범주 -> 변수
df = pd.get_dummies(df, columns=["범주"])
print(df)

from scipy.stats import sem
df = pd.DataFrame({
    "data": [1, 2, 3, 4, 5]
})
# 지원하지 않는 함수가 있다면 np를 붙여서 사용
df.agg(["mean", "var", "sem", "std", "skew", "kurt", "sum", "median", "max", np.max]).T

# 문제
data = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
# 이 데이터를 DataFrame으로 만들고 Categorical을 이용해서 범주화
# factorize를 이용해서 범주화 하시오

df = pd.DataFrame({"animal" : data})
df["animal_category"] = pd.Categorical(df["animal"]).astype("category")
print(df)
# 문자 : str, 날짜 : dt, 범주 : cat
print(df["animal_category"].cat.codes)

codes, uniques = pd.factorize(df["animal"])
df["animal_factorize"] = codes
print(df)

# 연속수치 범주화
# - cut(구간값), qcut(등분)
np.random.seed(42)
df = pd.DataFrame({
    "Age": np.random.randint(10, 80, 15),
    "Income": np.random.randint(1000, 10000, 15),
    "Score": np.random.randint(40, 100, 15),
    "BMI": np.round(np.random.uniform(15, 35, 15), 1)
})
print(df)
age_bins = [0, 18, 34, 64, 120]
age_labels = ["청소년", "청년", "중년", "노년"]
df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True)
print(df)

# 문제
# income을 저소득, 중저소득, 중고소득, 고소득으로 균등 범주화하시오
# income_bins = [0, 2500, 5000, 7500, 10000]
# income_labels = ["저소득", "중저소득", "중고소득", "고소득"]
# df["Income_Group"] = pd.cut(df["Income"], bins=income_bins, labels=income_labels, right=True)
df["Income_Group"] = pd.qcut(df["Income"], q=4, labels=["저소득", "중저소득", "중고소득", "고소득"])
print(df)

bmi_bins = [0, 18.5, 24.9, 29.9, 100]
bmi_labels = ["저체중", "정상", "과체중", "비만"]
df["BMI_Group"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=True)
print(df)

score_bins = [0, 59, 69, 79, 89, 100]
score_labels = ["F", "D", "C", "B", "A"]
df["Grade"] = pd.cut(df["Score"], bins=score_bins, labels=score_labels, right=True)
print(df)

# 나이 시각화
plt.figure(figsize=(6,4))
# 빈도수
df["Age_Group"].value_counts().sort_index().plot(kind="bar")
plt.title("연령대 분포")
plt.xlabel("연령 그룹")
plt.ylabel("빈도수")
plt.show()

# 도수 frequency
stacked = pd.crosstab( # 범주형 변수와 범주형 변수의 교차표
    df["Age_Group"],
    df["BMI_Group"]
)
print(stacked.head())
stacked.plot(
    kind="bar",
    stacked=True,
    figsize=(7,5)
)
plt.title("연령대별 BMI 분포")
plt.xlabel("연령대")
plt.ylabel("BMI")
plt.show()

import seaborn as sns
titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.columns)
# Index(['survived',  생존여부
#        'pclass',    객실등급
#        'sex',       성별
#        'age',       나이
#        'sibsp',     형제자매나 배우자수
#        'parch',     부모자 자녀수
#        'fare',      운임
#        'embarked',  퀸스타운 승선여부
#        'class',     등급
#        'who',
#        'adult_male',
#        'deck',
#        'embark_town',
#        'alive',     생존
#        'alone'],    혼자
#       dtype='object')

grouped = titanic.groupby(['sex', 'pclass'])['survived'].mean().reset_index()
print(grouped)
# 계층적 인덱스

sns.catplot(x='pclass', y='survived', hue="sex", kind="bar", data=grouped)
sns.catplot(x='pclass', y='survived', kind='bar', data=grouped)

data = {'col1':[0,1,2,3,4],
        'col2':[5,6,7,8,9],
        'level0': ['A', 'A', 'A', 'B', 'B'],
        'level1': ['X', 'X', 'Y', 'Y', 'Z'],
        'level2': ['a', 'a', 'b', 'c', 'a']}
df = pd.DataFrame(data=data)
df = df.set_index(['level0', 'level1', 'level2'])
print(df)

print(df.xs(key=('A', 'X')))

print(df.xs(key=('A')))

df.xs(('A', 'a'), level=[0, 'level2'])

print(df.stack())

print(df.swaplevel())

print(df.swaplevel(0))
print(df.swaplevel(i=0, j=1))

print(df.droplevel(axis=0, level=0))

arrays = [['학교1', '학교1', '학교1', '학교1', '학교2', '학교2', '학교3', '학교3'],
          [  '1반',   '2반',   '1반',   '2반',   '1반',   '2반',   '1반',  '2반']]

index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame({'합격': [1, 1, 1, 1, 2, 2, 3, 3],
                   '등수': np.arange(8)},
                  index=index)
df

grouped = df.groupby(level=0)
print(grouped)
grouped.sum() # 학교별 합계

df.groupby(level=1).sum() # 반으로 합계

sns.catplot(x='second', y='합격', kind='bar', data=df)
plt.title('반별 합격 데이터')
plt.show()
# 평균의 변동성 (표준오차를 고려해서)
# 평균출력

df.groupby(level=[0,1]).sum()

sns.catplot(x='first', y='합격', hue='second', kind='bar', data=df)
plt.title('학교별 반별 합격 데이터')
plt.show()

# 시계열 데이터
# range (list), arange (ndarray), date_range(index)
# pandas는 index 발달
# RangeIndex 디폴트 인덱스

index = pd.date_range('10/1/1999', periods=1100)

ts_origin = pd.Series(np.random.normal(0.5, 2, 1100), index)
print(ts_origin.shape)
print(ts_origin.head())
print(ts_origin.index)

# 이동 평균법 (Moveing Average), window 100
ts = ts_origin.rolling(window=100, min_periods=100).mean().dropna()

print(ts.shape)
print(ts.head())
ts.tail()

grouped = ts.groupby(lambda x: x.year)
grouped.mean()

# 연도별로 z-score를 생성 -> 데이터를 표준화
# 연도별로 평균=0, 표준편차=1 을 기준으로 변환
# 연도별로 각기 다른 평균과 표준편차를 적용
transformed = (ts.groupby(lambda x: x.year).transform(
    lambda x: (x - x.mean()) / x.std()
))
transformed

# transformed 이동평균법적용 -> z점수 표준화
compare = pd.DataFrame({'Original': ts_origin, 'Transformed': transformed})
compare.plot() # original은 고주파 -> 저주파

ts = ts_origin.rolling(window=20, min_periods=20).mean().dropna()
print(ts.shape)
transformed2 = (ts.groupby(lambda x: x.year).transform(lambda x: (x - x.mean()) / x.std()))
transformed2

compare = pd.DataFrame({'Original': ts_origin, 'Transformed2': transformed2, 'Transformed': transformed})
compare.plot()
# 20일 이평, 120일 이평
# 20일 이평이 120일 이평을 뚫고 내려가면 값이 하락할 징조
# 반대면 올라갈 징조

sf = pd.Series([1, 1, 2, 3, 3, 3])
sf.groupby(sf).filter(lambda x: x.sum() > 2)

# 문제
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['1반', '2반', '3반', '1반', '2반', '3반'],
                   '국어': rng.randint(80, 100, 6),
                   '수학': rng.randint(90, 100, 6)},
                   columns = ['key', '국어', '수학'])
df
# 반별로 최소값, 중위수, 최대값을 구해보시오
# df.groupby('key').min()
# df.groupby('key').median()
# df.groupby('key').max()
df.groupby('key').agg(['min', 'median', 'max'])

# 열별로 국어는 min, 수학은 max를 구하시오
# df.groupby('key')['국어'].min()
# df.groupby('key')['수학'].max()
df.groupby('key').agg({'국어': 'min', '수학': 'max'})

# 반별로 수학 표준편차가 1보다 큰 경우만 필터링하시오
df.groupby('key').filter(lambda x: x['수학'].std() > 1)

# long 포맷으로 보기
print(df.stack()) # 계층적

print(df.melt()) # long

# wide
df=pd.DataFrame({'key1': ['영업부', '영업부', '관리부', '관리부', '영업부'],
                 'key2'   : ['남', '여', '남', '여', '남'],
                 'data1'  : np.random.randn(5),
                 'data2'  : np.random.randn(5)})
print(df)

print(df.stack())

# 문제
# 성별 합계를 구하시오
df.groupby('key2').sum()[['data1', 'data2']]
# 부서별 성별 합계를 구하시오
df.groupby(['key1', 'key2']).sum()

from google.colab import drive
drive.mount('/content/drive')

# crosstab은 교차표 (series를 대상)
df = pd.read_csv('/content/drive/MyDrive/AI빅데이터_파이썬/dataset/jumsu.csv',
                 names=['Name', 'Exam', 'Subject', 'Result', 'Jumsu'],
                 encoding='utf-8')
df

# 행열 합계 : margins
pd.crosstab(df.Subject, df.Result, margins=True)

# 과목별로 대분류하고 시험별로 소분류된 패스 여부를 카운트하고 합계를 구하시오
pd.crosstab([df.Subject, df.Exam], df.Result, margins=True)

pd.crosstab(df.Subject, df.Exam, df.Jumsu, aggfunc=[np.mean], margins=True)

from scipy import stats
df = pd.DataFrame({
    "성별": ["남성", "여성", "남성", "여성", "남성"],
    "만족도": ["만족", "만족", "불만족", "불만족", "만족"]
})
crosstab = pd.crosstab(df["성별"], df["만족도"])
print(crosstab)

# 파이썬은 멀티 리턴 가능
# 독립성 검증 : 만족도에 성별 차이가 있는가
# 귀무가설 : 성별 만족도 차이가 없다
# 대립가설 : 성별 만족도 차이가 있다
# 귀무가설을 기각할 수 없다
chi2, p, dof, expected = stats.chi2_contingency(crosstab, correction=False)
print("카이제곱 통계량:", chi2)
print("p-값:", p)
print("자유도:", dof)
print("기대치:", expected)

# 카이제곱 통계량을 수식으로 계산하시오
((2 - 1.8)**2 / 1.8) + ((1 - 1.2)**2 / 1.2) + ((1 - 1.2)**2 / 1.2) + ((1 - 0.8)**2 / 0.8)

df = pd.DataFrame({"foo": ['one', 'one-1', 'two', 'two'],
                    "bar": ['A', 'A', 'B', 'C'],
                    "baz": [1, 2, 3, 4]})
df

# wide 데이터를 배치하는 pivot
df.pivot(index='foo', columns='bar', values='baz')

data = {
    "도시": [  "서울", "서울", "서울", "부산", "부산", "부산", "인천", "인천"   ],
    "연도": [  "2015", "2010", "2005", "2015", "2010", "2005", "2015", "2010"    ],
    "인구": [  9904312, 9631482, 9762546, 3448737, 3393191, 3512547, 2890451, 2632035  ],
    "지역": [  "수도권", "수도권", "수도권", "경상권", "경상권", "경상권", "수도권", "수도권"    ]
}
df1 = pd.DataFrame(data)
df1

# 데이터 중복을 불허
df1.pivot(index="도시", columns="연도", values="인구")

path='/content/drive/MyDrive/AI빅데이터_파이썬/dataset/sales-funnel.xlsx'
df = pd.read_excel(path)
df.head()
# 영업내역
# Representative 대표자

df["Status"].unique()

df["Status"] = df["Status"].astype("category")
df["Product"] = df["Product"].astype("category")
df.dtypes

# 매니저별 판매 현황 (중복)
# pivot_table 중복이 가능 -> 집계함수
pd.pivot_table(df, index=["Name"], columns=["Manager"], values=["Price"])

pd.pivot_table(df, index=["Name"], columns=["Manager"], values=["Price"], aggfunc=np.sum)

# 금액하고 개수를 보고 싶고, (금액평균, count)
pd.pivot_table(df, index=["Name"], columns=["Manager"],
               values=["Price", "Quantity"],
               aggfunc=[np.mean, len],
               fill_value=0, margins=True)

pd.pivot_table(df, index=["Manager", "Rep"], values=["Price"], aggfunc=np.sum)

table = pd.pivot_table(df, index=["Manager", "Rep"],
               columns=["Product"],
               values=["Price", "Quantity"],
               aggfunc=[np.mean, len],
               fill_value = 0,
               margins=True)
table

table = pd.pivot_table(df, index=["Manager", "Status"],
               columns=["Product"],
               values=["Price", "Quantity"],
               aggfunc={"Quantity":len, "Price":["sum", "mean"]},
               fill_value = 0,
               observed=False)
table

table.query('Manager == ["Debra Henley"]')

table.query('Status == ["pending", "won"]')

plt.figure(figsize=(12, 8))
sns.heatmap(table['Quantity'], annot=True, cmap='YlGnBu')
plt.title('매니저, 상태, 상품별 개수')
plt.show()

# 매니저별 상태별 평균가격을 시각화하시오
# plt.figure(figsize=(12, 8))
# sns.heatmap(table.groupby(['Manager', 'Status'])['Price'].mean(), annot=True, cmap='YlGnBu')
# plt.title('매니저별 상태별 평균가격')
# plt.show()
sns.barplot(x="Manager", y="Price", hue="Status", data=df)
plt.title('매니저별 상태별 평균가격')
plt.show()

# 상품별로 총매출액과 개수를 출력해 보시오
# table = pd.pivot_table(df, index=["Product"],
#                values=["Price", "Quantity"],
#                aggfunc={"Price":np.sum, "Quantity":np.sum},
#                fill_value = 0,
#                margins=True)
# table

product_sales = pd.DataFrame({
    '총 매출액': table['Price', 'sum'].sum(),
    '개수': table['Quantity', 'len'].sum()
})

tips = sns.load_dataset("tips")
tips.tail()

tips.dtypes

# 남녀인지, 흡연여부에 따른 팁금액관계를 확인하시오
tips.groupby(["sex", "smoker"])[["tip"]].agg(
    lambda x: x.max() - x.min()
)

tips.groupby(["sex", "smoker"])[["tip"]].agg(
    lambda x: x.mean()
)

pivot_table=tips.pivot_table(index=['sex'],
                             columns=['smoker'], values='tip', aggfunc=np.mean,
                             margins=False, fill_value=0)
pivot_table

# 문제
# 팁의 비율이 요일과 점심/저녁 여부, 인원 수에 따라 어떤 영향을 받는지 알아보시오
# 또 카이제곱분석을 통해서 유의미한 차이가 있는지 확인하시오

pivot_table = tips.pivot_table(index=['day', 'time'],
                               columns=['size'],
                               values='tip',
                               aggfunc="mean",
                               margins=True,
                               fill_value=0,
                               observed=False)
pivot_table

chi2, p, dof, expected = stats.chi2_contingency(pivot_table)
print("카이제곱 통계량:", chi2)
print("p-값:", p)
print("자유도:", dof)
print("기대치:", expected)
# 귀무가설을 기각하지 못함

# 문제
# 흡연자 비율을 구하시오
# len(tips[tips["smoker"] == "Yes"]) / len(tips)
print("흡연비율", tips["smoker"].value_counts(normalize=True))
# 요일별 평균 팁을 구하시오
print("평균팁", tips.groupby("day")["tip"].mean())
# Dinner와 Lunch 중 어느 쪽이 매출이 더 큰가?
print("시간매출", tips.groupby("time")["total_bill"].sum())
# tip 비율을 계산하시오
print("팁비율", tips["tip"] / tips["total_bill"])
# 성별과 흡연 여부의 교차표를 작성하시오
table = pd.crosstab(tips["sex"], tips["smoker"])
print(table)

from sklearn.datasets import load_iris
from scipy import stats
iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)
df["species"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)
print(df.head())

x = df["sepal_length"]
print("평균:", x.mean()) # series
print("분산:", x.var())
print("표준편차:", x.std())
print("왜도:", stats.skew(x)) # 함수의 매개변수로 전달
print("첨도:", stats.kurtosis(x))
print("표준오차 SEM:", stats.sem(x))    # 표준편차 / sqrt(n)
print("변동계수:", stats.variation(x))  # 표준편차 / 평균
print("scipy describe:", stats.describe(x))

mu, sigma = stats.norm.fit(x) # 정규분포 평균, 표준편차 추정
print("\n추정 평균:", mu)
print("추정 표준편차:", sigma)
value = 6.0
# 밀도함수 x축은 표준편차, y축은 밀도 (연속된 데이터)
pdf_value = stats.norm.pdf(value, loc=mu, scale=sigma)
print("\nPDF:", pdf_value)

# 누적확률
cdf_value = stats.norm.cdf(value, loc=mu, scale=sigma)
print("CDF P(X <= 6.0):", cdf_value)

# sf 생존함수 = 1-cdf
sf_value = stats.norm.sf(value, loc=mu, scale=sigma)
print("SF P(X > 6.0):", sf_value)

# 분위수
q_95 = stats.norm.ppf(0.95, loc=mu, scale=sigma)
print("95% 분위수:", q_95)

sample = stats.norm.rvs(loc=mu, scale=sigma, size=10, random_state=42)
print("정규분포 난수:", sample)

# 신뢰구간 : 평균 - 1.96 * sem(표준오차) < 구간 < 평균 + 1.96 * sem(표준오차)
n = len(x)
sem = stats.sem(x)
ci_95 = stats.t.interval( # t 분포에서의 신뢰구간
    confidence=0.95,
    df=n-1,
    loc=x.mean(),
    scale=sem
)
print("\n95% 신뢰구간:", ci_95)

# ANOVA 분석 (분산분석) - 평균분석
setosa = df[df["species"] == "setosa"]["petal_length"]
versicolor = df[df["species"] == "versicolor"]["petal_length"]
virginica = df[df["species"] == "virginica"]["petal_length"]
f_stat, pvalue = stats.f_oneway(setosa, versicolor, virginica)
print("ANOVA 결과")
print("F-statistic:", f_stat)
print("p-value:", pvalue)

# 회귀분석
x = df["petal_length"]
y = df["petal_width"]

result = stats.linregress(x, y)
print("slope:", result.slope)
print("intercept:", result.intercept)
print("결정계수:", result.rvalue)
print("pvalue:", result.pvalue)
print("stderr:", result.stderr)

# log 변환 -> 정규분포 변환
# BoxCox 변환 -> 양수만 대상 -> 고급 (람다를 계산)
# 람다 : 얼마나 변화시키면 정규분포에 더 가까워지는가
x = df["petal_length"]
x_boxcox, lambda_value = stats.boxcox(x)
print("lambda:", lambda_value)
print("변환 전 skew:", stats.skew(x))
print("변환 후 skew:", stats.skew(x_boxcox))
best_lambda = stats.boxcox_normmax(x)
print("최적 lambda:", best_lambda)

# 브라우저 -> 파싱 -> 렌더링 (화면)
# !pip install BeautifulSoup4

# <div>
import requests # html 접속객체
from bs4 import BeautifulSoup

class HTMLParser:
    def parse_url(self, url):
        response = requests.get(url)
        # 의미있는 단어들로 변환
        soup = BeautifulSoup(response.text, "html.parser")
        return [(table('id'), self.parse_html_table(table))
                   for table in soup.find_all('table')]
        # 테이블의 id들을 리스트로 리턴
    def parse_html_table(self, table): # 테이블 파싱함수
        n_columns = 0 # 행과 열로 표현되는 테이블
        n_rows=0
        column_names = [] # 컬럼이름
        for row in table.find_all('tr'):
            td_tags = row.find_all('td')
            if len(td_tags) > 0: # 내용이 있으면
                n_rows+=1 # 행 증가
                if n_columns == 0: # 열을 처음으로 지정하면
                    n_columns = len(td_tags) # 열수 확인
            th_tags = row.find_all('th') # 헤더
            # 헤더를 찾고, 컬럼이름이 정해지지 않았으면
            if len(th_tags) > 0 and len(column_names) == 0:
                for th in th_tags:
                    column_names.append(th.get_text()) # 열이름을 확인
        if len(column_names) > 0 and len(column_names) != n_columns:
            raise Exception("컬럼 타이틀이 컬럼 숫자와 맞지 않는다. ")
        # 컬럼이름이 있으면 컬럼이름으로 아니면 숫자 인덱스로
        columns = column_names if len(column_names) > 0 else range(0,n_columns)
        df = pd.DataFrame(columns = columns,index= range(0,n_rows))
        # 데이터를 채워주는 과정
        row_marker = 0 # 행을 가리키는 지시자
        for row in table.find_all('tr'): # 행의 시작
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                # 웹에서는 쓸데없는 텍스트가 많이 있음
                re_text=column.get_text().replace("\n",'')
                re_text=re_text.replace("\t",'')
                re_text=re_text.replace("\r",'')
                df.iloc[row_marker,column_marker] = re_text
                column_marker += 1
            if len(columns) > 0:
                row_marker += 1 # 행증가
        for col in df: # 열별처리 (DF은 열중심)
            try: # 웹에서는 전부 텍스트
                df[col] = df[col].astype(float) # 숫자는 숫자로 변환
            except ValueError:
                pass
        return df

date = "2021.04.02"
url_part1 = """http://dart.fss.or.kr/dsac001/search.ax?selectDate="""
url_part2 = """&sort=&series=&mdayCnt=0&currentPage="""
url = url_part1 + date + url_part2
print(url)
hp = HTMLParser()
table = hp.parse_url(url)[0][1]
table

url = 'https://www.w3schools.com/html/html_table_styling.asp'
print(url)
hp = HTMLParser()
tables = hp.parse_url(url)
print(tables)

import requests # html 접속객체
from bs4 import BeautifulSoup

class HTMLParser:
    def find_title_for_table(self, table, index, soup=None):
        # 1) table 자체 id
        if table.get('id'):
            return table.get('id')

        # 2) table 안 caption
        caption = table.find('caption')
        if caption:
            return caption.get_text(strip=True)

        # 3) 바로 이전 형제 태그 중 h태그만
        for sibling in table.previous_siblings:
            if not hasattr(sibling, 'name'):
                continue  # 텍스트 노드 스킵
            if sibling.name in ['div','h1','h2','h3','h4','h5','h6']:
                return sibling.get_text(strip=True)
            break  # h태그 아닌 다른 태그 나오면 바로 중단

        # 4) 사이트 전용 — soup 전체에서 탐색
        if soup:
            tag = soup.select_one('h4.tit-page')
            if tag:
                return tag.get_text(strip=True)

        # 5) 못 찾으면 인덱스
        return f"table_{index}"

    def parse_url(self, url):
        response = requests.get(url)
        # 의미있는 단어들로 변환
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for i, table in enumerate(soup.find_all('table')):
            title = self.find_title_for_table(table, i, soup)
            results.append((title, self.parse_html_table(table)))

        return results

        # 테이블의 id들을 리스트로 리턴
    def parse_html_table(self, table): # 테이블 파싱함수
        n_columns = 0 # 행과 열로 표현되는 테이블
        n_rows=0
        column_names = [] # 컬럼이름
        for row in table.find_all('tr'):
            td_tags = row.find_all('td')
            if len(td_tags) > 0: # 내용이 있으면
                n_rows+=1 # 행 증가
                if n_columns == 0: # 열을 처음으로 지정하면
                    n_columns = len(td_tags) # 열수 확인
            th_tags = row.find_all('th') # 헤더
            # 헤더를 찾고, 컬럼이름이 정해지지 않았으면
            if len(th_tags) > 0 and len(column_names) == 0:
                for th in th_tags:
                    column_names.append(th.get_text(strip=True)) # 열이름을 확인 # strip=True로 공백 제거
        if len(column_names) > 0 and len(column_names) != n_columns:
            raise Exception("컬럼 타이틀이 컬럼 숫자와 맞지 않는다. ")
        # 컬럼이름이 있으면 컬럼이름으로 아니면 숫자 인덱스로
        columns = column_names if len(column_names) > 0 else range(0,n_columns)
        df = pd.DataFrame(columns = columns,index= range(0,n_rows))
        # 데이터를 채워주는 과정
        row_marker = 0 # 행을 가리키는 지시자
        for row in table.find_all('tr'): # 행의 시작
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                # 웹에서는 쓸데없는 텍스트가 많이 있음
                re_text=column.get_text().replace("\n",'')
                re_text=re_text.replace("\t",'')
                re_text=re_text.replace("\r",'')
                df.iloc[row_marker,column_marker] = re_text
                column_marker += 1
            if len(columns) > 0:
                row_marker += 1 # 행증가
        for col in df: # 열별처리 (DF은 열중심)
            try: # 웹에서는 전부 텍스트
                df[col] = df[col].astype(float) # 숫자는 숫자로 변환
                if (df[col] % 1 == 0).all():
                  df[col] = df[col].astype(int)
            except ValueError:
                pass
        return df

# 과제 : 테이블을 크롤링해서 출력하시오.
# 대상은 자유롭게 선정한 다음 구현하시오.
url = 'https://www.koreabaseball.com/Record/TeamRank/TeamRankDaily.aspx'
print(url)
hp = HTMLParser()
tables = hp.parse_url(url)

title, df = tables[0]
print(df.dtypes)

print(f"\n[테이블 제목: {title}]")
display(df.style.hide(axis='index').format({
    '승률': '{:.3f}',
    '게임차': '{:.1f}'
}))