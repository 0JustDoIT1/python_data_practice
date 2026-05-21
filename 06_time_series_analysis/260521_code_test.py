from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np

# sales데이터를 로딩하고 다음을 처리하시요
sales = pd.read_csv('/content/drive/MyDrive/AI빅데이터_파이썬/dataset/sales.csv')
sales.info()
sales.dtypes
sales.head()

# 'Customer Number' 를 int형으로 처리하시요
sales['Customer Number'].unique()
sales['Customer Number'] = sales['Customer Number'].astype('int')
sales['Customer Number'].dtypes

# df['Jan Units'] 를 int형으로 처리하시요
sales['Jan Units'].unique()
sales['Jan Units'] = pd.to_numeric(sales['Jan Units'], errors='coerce')
sales = sales.dropna(subset=['Jan Units'])
sales['Jan Units'] = sales['Jan Units'].astype('int')
sales['Jan Units'].dtypes

# '2016' 데이터를 float형으로 처리하시요
sales['2016'].unique()
sales['2016'] = sales['2016'].str.replace('$', '', regex=False)
sales['2016'] = sales['2016'].str.replace(',', '', regex=False)
sales['2016'] = sales['2016'].astype('float')
sales['2016'].dtypes

# 숫자 데이터에 들어 있는 ,와 $를 제거하시요
sales['2017'] = sales['2017'].str.replace('$', '', regex=False)
sales['2017'] = sales['2017'].str.replace(',', '', regex=False)
sales['2017'] = sales['2017'].astype('float')
sales['2017'].dtypes

# 수치형 데이터중 %를 수치로 바꾸시요 (문자를 떼고 100으로 나누기 )
sales.head()
sales['Percent Growth'] = sales['Percent Growth'].str.replace('%', '', regex=False)
sales['Percent Growth'] = sales['Percent Growth'].astype('float')
sales['Percent Growth'] = sales['Percent Growth'] / 100
sales['Percent Growth'].dtypes

# Active를 True / False로 변경하시요 ( boolean값 ) : Y->1, N->0
sales.head()
# sales['Active'] = sales['Active'].apply(lambda x: 1 if x == 'Y' else 0)
sales['Active'] = sales['Active'].map({'Y': 1, 'N': 0})

sales