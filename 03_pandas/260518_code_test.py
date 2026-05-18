# Bunch 데이터 -> 그 데이터를 duckdb에 저장
# 똑같은 작업을 duckdb 쿼리를 이용해서 데이터를 가지고 온 다음
# dataframe 작업으로 시각화하시오

from google.colab import drive
drive.mount('/content/drive')

# Bunch
import pandas as pd
import numpy as np
# duckdb 설치 및 연결
import duckdb
conn = duckdb.connect('iris.db') # 파일로 저장

from sklearn.datasets import load_iris
iris = load_iris()
type(iris)
iris['data']
iris['target'] # 종속변수
iris['feature_names']
# column
iris = pd.DataFrame(data = np.c_[iris['data'], iris['target']], columns = iris['feature_names'] + ['target'])
iris.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
iris['Name'] = iris['Name'].astype('int32')
iris

# duckdb에 저장
conn.sql("""
  CREATE TABLE IF NOT EXISTS iris AS SELECT * FROM iris
""")

# duckdb 테이블 목록 확인
conn.sql("SHOW TABLES")

# duckdb에서 꺼내오기
irisdb = conn.sql("""
  SELECT * FROM iris
""").df()

irisdb

# iris.info()
conn.sql("""
  DESCRIBE iris
""")
# iris.head()
conn.sql("""
  SELECT * FROM iris LIMIT 5
""")
# iris.plot()
irisdb.plot()