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
    style="white",
    rc={
        "font.family": "NanumGothic",
        "axes.unicode_minus": False,
    }
)

# 확률 모델
# 몬테칼로를 이용해서 pi값을 구하시오
# 반지름이 1인 원

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
num_samples = 10000
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
x_samples = np.random.uniform(x_min, x_max, num_samples)
y_samples = np.random.uniform(y_min, y_max, num_samples)
# 원 내부 여부
is_inside = x_samples**2 + y_samples**2 <= 1
# 전체 사각형 넓이
box_area = (x_max - x_min) * (y_max - y_min)
# pi 근사
pi_estimate = box_area * (np.sum(is_inside) / num_samples)

# 원 내부 점
plt.scatter(
    x_samples[is_inside],
    y_samples[is_inside],
    color='blue',
    s=1,
    alpha=0.5,
    label='Inside Circle'
)

# 원 외부 점
plt.scatter(
    x_samples[~is_inside],
    y_samples[~is_inside],
    color='red',
    s=1,
    alpha=0.3,
    label='Outside Circle'
)

# 실제 원 테두리
theta = np.linspace(0, 2*np.pi, 500)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

plt.plot(x_circle, y_circle, linewidth=2, color='black')

plt.title(f"몬테칼로 pi값 : {pi_estimate:.6f}")
plt.xlim(-1,1)
plt.ylim(-1,1)

# 원 비율 유지
plt.gca().set_aspect('equal')

plt.legend()
plt.grid(True)
plt.show()