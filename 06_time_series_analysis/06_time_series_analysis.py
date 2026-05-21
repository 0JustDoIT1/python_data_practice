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

from google.colab import drive
drive.mount('/content/drive')

# range, to_datetime 함수
import pandas as pd
# B: 영업일, Y: 연, M: 월, D: 일, Q: quarter 분기, H : 시간, S: 초
dt_index = pd.date_range("2025-01-01", periods=5, freq="B")
pr_index = pd.period_range("2025-01", periods=4, freq="M")
td_index = pd.to_timedelta([0, 1, 2], unit="D") # 간격

print("DatetimeIndex:", dt_index)
print("PeriodIndex:", pr_index)
print("TimedeltaIndex:", td_index)

# UTC 세계표준시 : 그리니치 천문대를 기준
import pytz
from datetime import datetime, timezone, timedelta
naive_datetime = datetime(2025, 2, 25, 10, 0, 0)
tz_seoul = timezone(timedelta(hours=9)) # KSC 서울기준
# 시간 변환
aware_datetime = pytz.utc.localize(naive_datetime).astimezone(tz_seoul)
aware_datetime

# parsing
str_time = "2023-07-20 17:31:49" # 문자열
time = datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
print(time)

# 외부 입력 시간 변환
# nano초까지 지원 datetime64
import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64)
print(date)

tm=pd.Timestamp(year=2026, month=5, day=21, hour=12)
tm.day_of_week # 월요일을 0 -> 0, 1, 2, 3, 4

type(tm) # 정확한 시간 표현

tm.to_numpy()

# 날짜연산
import datetime
now = datetime.datetime.now()
df = pd.DataFrame({"value": [1, 2, 3]},
                  index=[now, now + pd.Timedelta(days=1),
                         now + pd.Timedelta(days=2)])
df

pd.timedelta_range(start='1 day', periods=4)

pd.timedelta_range(start='1 day', end='2 days', freq='6H')

range_date = pd.date_range(start = '1/1/2019',
                           end = '1/2/2019',
                           freq='Min')
print(range_date)

date_index = pd.date_range('2015-07-03', '2015-07-10')
# 행인덱스
data = pd.Series(range(1,9), index=date_index)
data

data['2015-07-04':'2015-07-05']

data['2015-07']

data['2015']

idx = pd.date_range('2018-01-01', periods=5, freq='H')
ts=pd.Series(range(len(idx)), index=idx)
ts

# groupby 그룹핑 처리
# resample은 groupby처럼 -> 집계 함수 필요
ts.resample('2H').mean() # 2개의 시간을 한개

# 빈데이터는 선형 보간법으로 채움
ts.resample('5min').interpolate(method='linear')

ts.resample('5min').interpolate(method='nearest')

ts.resample('5min').interpolate(method='spline', order=3)

ts.resample('5min').bfill()

# T : minutes
import numpy as np
rng =pd.date_range('1/1/2014', periods=120, freq='T')
ts=pd.Series(np.arange(120), index = rng)
print(ts)

# open: 개장가, high: 고가, low: 저가, close: 종가
print(ts.resample('5min').ohlc())

print(ts.groupby(lambda x: x.month).mean())

# 월말 기준
np.random.seed(0)
ts = pd.Series(np.random.randn(4), index=pd.date_range(
    "2019-1-1", periods=4, freq="ME"
))
ts

ts.shift(-1)

# 변화량
ts.shift(-1) - ts
# 뒤에서 앞을 뺌 -> 정방향 변화량과 변화율

# 단순평균
s = pd.Series([2, 3, 4, 10, 11, 18])
s.rolling(3, center=True).mean()

df=pd.DataFrame({
    "a": [1, 2, 3, 4, 5, 6],
    "b": [1, 1, 2, 3, 5, 8]
})
print(df)
df.diff() # 차분 (다음 것에서 이전 것을 뺌 (등분))

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
np.random.seed(42)
n = 220
t = np.arange(n) # 시간
trend =0.08 * t # 추세
seasonal = 2.0 * np.sin(2 * np.pi * t / 7) # 계절 변동 (7일 변동)
noise = np.random.normal(0, 0.6, size=n)
y = trend + seasonal + noise

idx = pd.date_range("2025-01-01", periods=n, freq="D")
df = pd.DataFrame({"y": y}, index=idx)

df["y_lag1"] = df["y"].shift(1) # 빼는 순서만 변경
df["diff_1"] = df["y"].diff(1)
df["y_lag7"] = df["y"].shift(7)
df["diff_7"] = df["y"].diff(7) # 7 lag 간의 차분

preview = df[["y", "y_lag1",
              "diff_1", "y_lag7", "diff_7"]].head(12).copy()
print("Shift/Lag/차분 미리보기", preview)

# 1주기 차분한 결과 시각화 - 추세가 제거
plt.figure(figsize=(10,4))
plt.plot(df.index, df["y"], marker="o", linestyle="-")
plt.title("원 시계열 (추세 + 주기 + 잡음)")
plt.xlabel("날짜")
plt.ylabel("값")
plt.show()

# 원시계열 데이터의 acf(auto corelation function) 그래프
plt.figure(figsize=(8,4))
plot_acf(df["y"].dropna(), lags=30)
plt.title("ACF - 원 시계열 (7일 주기 뚜렷)")
plt.show

plt.figure(figsize=(8,4))
plot_acf(df["diff_1"].dropna(), lags=30)
plt.title("ACF -  1차 차분 (추세 제거)")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(df.index, df["diff_7"], marker="o", linestyle="-")
plt.title("계절 차분: diff(7)-y[t]-y[t-7](주기 제거 효과)")
plt.xlabel("날짜")
plt.ylabel("차분 값")
plt.show()

plt.figure(figsize=(8,4))
plot_acf(df["diff_7"].dropna(), lags=30)
plt.title("ACF - 7일 차분 (계절성 제거)")
plt.show()

# acf(auto corelation function)가 MA(moving average) 차수결정
# 직접과 간접을 모두 포함
# pacf(partial auto corelation function)가 AR(auto regression) 차수결정
# 중간요소를 제거하고 직접적인 요인만 고려한 것

import yfinance as yf

gs_ticker = yf.Ticker("078930.KS")
gs = gs_ticker.history(start="2018-01-01", end="2020-03-06")
print(gs.head())

ma5 = gs['Close'].rolling(window=5).mean()
ma5.head(10) # 5일 이평

# 문제
# 1) 결측치를 확인하시오
# 2) 구해진 값의 5일 이평(ma5), 20일 이평(ma20), 60일 이평(ma60), 120일 이평(ma120)을 구하시오
# 3) 이를 변수로 추가하시오
gs.isna().sum()
full_range = pd.date_range(gs.index.min(), gs.index.max(), freq="D")
gs.dtypes

# 이동평균법
new_gs = gs[gs['Volume'] != 0]
ma5 = new_gs['Close'].rolling(window=5).mean()
ma20 = new_gs['Close'].rolling(window=20).mean()
ma60 = new_gs['Close'].rolling(window=60).mean()
ma120 = new_gs['Close'].rolling(window=120).mean()

new_gs["ma5"] = ma5
new_gs["ma20"] = ma20
new_gs["ma60"] = ma60
new_gs["ma120"] = ma120

new_gs.tail(5)

new_gs.plot()

new_gs[["ma5", "ma20", "ma60", "ma120"]].plot()

# !pip install mplfinance

import mplfinance as mpf
mpf.plot(gs, type='candle', mav=(3, 6, 9), volume=True)

# !pip install -U finance-datareader

import FinanceDataReader as fdr
df = fdr.DataReader('AAPL', '2025/10')

mpf.plot(df)

df=fdr.DataReader('USD/KRW', '1995')
df.isna().sum()
df = df.dropna()

mpf.plot(df)

df_spx = fdr.StockListing('S&P500')
df_spx.isna().sum()
df_spx = df_spx.dropna()
df_spx.index

df_spx.head()

hynix=fdr.DataReader('000660', '2020-01-01', '2026-04-30')
hynix['Close'].plot()

mpf.plot(hynix, type='candle', mav=(3, 6, 9), volume=True)

hynix=fdr.DataReader('000660', '2020-01-01', '2026-04-30')
hynix.head(10)
fig = plt.figure(figsize=(12, 8))
# 축분할
top_axes = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=4)
# 아래축
bottom_axes = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
# exponent 출력
bottom_axes.get_yaxis().get_major_formatter().set_scientific(False)
top_axes.plot(hynix.index, hynix['Close'], label='Adjusted Close') # 수정된 종가
bottom_axes.plot(hynix.index, hynix['Volume'])
plt.tight_layout()

# !pip install pymysql

# 문제
# hynix 데이터를 cloud에 테이블 생성하고 저장하시오
# 필드이름
# trade_date, Open, High, Low, Close, Volume, chg
# 저장된 데이터를 가지고 와서 날짜를 Dataframe의 인덱스로 지정하고
# 이를 시각화하시오

import pymysql

conn = pymysql.connect(
    host='104.198.27.181',
    user='hbc3869',
    password='Zmfjszl123!',
    database='daejeon',
    charset='utf8'
)
cur = conn.cursor()

hynix=fdr.DataReader('000660', '2020-01-01', '2026-04-30')
hynix.head(10)

# hynix = hynix.rename(columns={
#     'Open': 'Open',
#     'High': 'High',
#     'Low': 'Low',
#     'Close': 'Close',
#     'Volume': 'Volume',
#     'Change': 'chg'
# })
hynix = hynix.reset_index()
hynix = hynix.rename(columns={'Date': 'trade_date'})

create_sql = """
  CREATE TABLE IF NOT EXISTS hynix_stock (
    trade_date DATE NOT NULL PRIMARY KEY,
    Open DOUBLE,
    High DOUBLE,
    Low DOUBLE,
    Close DOUBLE,
    Volume BIGINT,
    chg DOUBLE
  )
"""

cur.execute(create_sql)

insert_sql = """
  INSERT INTO hynix_stock (trade_date, Open, High, Low, Close, Volume, chg)
  VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

for row in hynix.itertuples(index=False):
  cur.execute(insert_sql, tuple(row))

conn.commit()
print("테이블을 생성했습니다.")

select_sql = """
  SELECT * FROM hynix_stock
"""

# cur.execute(select_sql)

# result = cur.fetchall()

# print(result)

# columns = [desc[0] for desc in cur.description]
# print(columns)

# df = pd.DataFrame(result, columns=columns)
# df['trade_date'] = pd.to_datetime(df['trade_date'])
# df = df.set_index('trade_date')
# df.info()
# df.head()

df = pd.read_sql(select_sql, conn, parse_dates=['trade_date'])
df.set_index('trade_date', inplace=True)
df.info()
df.head()

target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[target_cols] = df[target_cols].apply(
    pd.to_numeric, errors="coerce"
)
df = df.dropna(subset=target_cols)
mpf.plot(df, type="candle", mav=(5, 20, 60), volume=True, style="yahoo")
conn.close()

# ORM : object relation mapper
# 장고 : 기본이 ORM임
# flask -> django -> react -> flutter

from sqlalchemy import create_engine
import mplfinance as mpf
hynix = fdr.DataReader('000660', '2020-01-01', '2024-12-31')
print(hynix.head())

engine = create_engine(
    "mysql+pymysql://hbc3869:Zmfjszl123!@104.198.27.181:3306/daejeon"
)
hynix.to_sql("hynix_stock", con=engine, if_exists="replace", index=True)
engine.dispose()

engine = create_engine(
    "mysql+pymysql://hbc3869:Zmfjszl123!@104.198.27.181:3306/daejeon"
)
df = pd.read_sql("SELECT * FROM hynix_stock", con=engine, parse_dates=['Date'])
df.set_index('Date', inplace=True)
print(df.head())
mpf.plot(df, type='candle', mav=(5, 20, 60), volume = True, style='yahoo')
engine.dispose()

# 자기 상관 테스트
from statsmodels.stats.stattools import durbin_watson
g = np.array([1, 2, 3, 1, 2, 3])
gfg = durbin_watson(g)
print(gfg)

from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv(
    '/content/drive/MyDrive/AI빅데이터_파이썬/dataset/AirPassengers.csv'
)
df.head()

# 강한 양의 자기 상관 있음
durbin_watson(df['AirPassengers'])

# 문제
# Month를 인덱스에 입력하시오
df['Month'] = pd.to_datetime(df['Month'])
df.index=df['Month']
df

del df["Month"]
df

df.plot()

decomposed = seasonal_decompose(df, model = 'addictive')
# 가법모형 : 시간에 따라서 변화가 작은 경우
# 승법모형 : 시간에 따라서 변화가 큰 경우

trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

plt.plot(df['AirPassengers'], label="original")
plt.plot(trend, label="trend")
plt.plot(seasonal, label="seasonal")
plt.plot(residual, label="residual")
plt.legend()
plt.show()

index = pd.date_range('10/1/1999', periods=1100)
ts_origin = pd.Series(np.random.normal(0.5, 2, 1100), index)
print(ts_origin.shape)
print(ts_origin.head())
ts = ts_origin.rolling(window=100, min_periods=100).mean().dropna()
print(ts.shape)
print(ts.head())
ts.tail()

# 문제 연도별 z점수를 구하시오
transformed = ts.groupby(lambda x: x.year).transform(
    lambda x: (x - x.mean()) / x.std()
)
transformed

compare = pd.DataFrame({'Original': ts_origin, 'Transformed': transformed})
compare.plot()

# 볼린저 밴드 (매매전법)
data = pd.read_csv('/content/drive/MyDrive/AI빅데이터_파이썬/dataset/stock_px.csv')
print(data.info())
prices = data['MSFT'] # 마이크로 소프트

window = 20
n_std = 2 # 2표준편차
ceneter_line = prices.rolling(window=window).mean()
upper_band = ceneter_line + n_std * prices.rolling(window=window).std()
lower_band = ceneter_line - n_std * prices.rolling(window=window).std()

signals = []
for i in range(len(prices)):
  if prices[i] > upper_band[i]:
    signals.append(1) # 판매시점
  elif prices[i] < lower_band[i]:
    signals.append(-1) # 구매시점
  else:
    signals.append(0)
print(signals)

compare = pd.DataFrame({
    'Original': prices, 'center_line': ceneter_line,
    'upper': upper_band, 'lower': lower_band
})
compare.plot()
# upper 표준편차 상 2배수
# lower 표준편차 하 2배수

# !pip install mpl_finance

import matplotlib.ticker as ticker
import mpl_finance as matfin
import datetime

start = datetime.datetime(2019, 3, 1) # 1개월 봉차트
end = datetime.datetime(2019, 3, 31)
hynix = fdr.DataReader('000660', start, end)
print(hynix.head(10))

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
day_list = range(len(hynix))
name_list = []
for day in hynix.index:
  name_list.append(day.strftime('%d'))
ax.xaxis.set_major_locator(ticker.FixedLocator(day_list))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))
matfin.candlestick2_ohlc(ax, hynix['Open'], hynix['High'], hynix['Low'], hynix['Close'],
                         width=0.5, colorup='r', colordown='b')
plt.show()

# 맥주
from sklearn.metrics import mean_squared_error
path = '/content/drive/MyDrive/AI빅데이터_파이썬/dataset/austr.csv'
df = pd.read_csv(path, parse_dates=["Month"])
print(df.head())

df.rename(columns={"Month":"month", "Monthly beer production": "M_beer_Prod"},
          inplace = True)

plt.figure(figsize=(15,5))
df["M_beer_Prod"].plot()

df["Day"]=df.month.dt.day
df["Year"]=df.month.dt.year
df["months"]=df.month.dt.month

df.groupby("Year")["M_beer_Prod"].mean().plot.bar()

df['Month']=pd.to_datetime(df['month'])
df.index=df['Month']
df

df.Timestamp=pd.to_datetime(df.month)
df.index=df.Timestamp

df=df.resample("ME").mean() # month end 말일기준
df.head()

len(df)*0.7

train=df.iloc[0:333]
valid=df.iloc[333:] # 30%

import statsmodels.api as sm
sm.tsa.seasonal_decompose(train["M_beer_Prod"]).plot()

# 단순 지수 이동 평균법
# 단순 MA, weighted MA, simple EMA, EMA

from statsmodels.tsa.api import SimpleExpSmoothing
y_hat=valid.copy()
# Smoothing -> noise를 제거
fit=SimpleExpSmoothing(np.asarray( # 데이터를 학습
    train["M_beer_Prod"]
)).fit(smoothing_level=0.8, optimized=False)
y_hat["SES"]=fit.forecast(len(valid)) # 모델 예측
plt.figure(figsize=(15,8))
plt.plot(train["M_beer_Prod"], label="Train")
plt.plot(valid["M_beer_Prod"], label="Valid")
plt.plot(y_hat["SES"], label="Simple Exponential Smoothing")
plt.legend(loc="best")
plt.show()

from statsmodels.tsa.api import ExponentialSmoothing
y_hat=valid.copy()
# 최대 우도법으로 확률 모델을 구성
# additive 가법으로
fit2=ExponentialSmoothing(np.asarray(train["M_beer_Prod"]),
                         trend='add', 
                         damped_trend=True, 
                         seasonal='add', 
                         seasonal_periods=12).fit()
              
y_hat['HS'] = fit2.forecast(len(valid))
plt.figure(figsize=(15,8))
plt.plot(train["M_beer_Prod"], label="Train")
plt.plot(valid["M_beer_Prod"], label="Valid")
plt.plot(y_hat['HS'], label="Exponential Smoothing")
plt.legend(loc="best")
plt.show()