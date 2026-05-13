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
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'})

# !ls -l /usr/share/fonts/truetype/nanum/

# !fc-list | grep -i "nanum"

print(f"현재 설정된 폰트: {plt.rcParams['font.family']}")

print(f"폰트 설정: {plt.rcParams['font.family']}")
# dot for inch 인치당 dot 수 => 프린터할 때의 기준 : 100
print(f"해상도(DPI): {plt.rcParams['figure.dpi']}")
plt.plot([-1,0,1], [1,0,1])
plt.title("현재 디렉토리 설정 적용 테스트")
plt.xlabel("x 축 (한글)")
plt.show()

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 300

font_path_myeongjo = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
fe_myeongjo = fm.FontEntry(fname=font_path_myeongjo, name='MyNanumMyeongjo')
fm.fontManager.ttflist.insert(0, fe_myeongjo)

# 상태 기반
import numpy as np
def draw_with_font(font_name, title_text):
  plt.rcParams.update({'font.family': font_name,
                       'font.size': 15, 'axes.unicode_minus': False})
  x = np.linspace(0, 10, 10)
  y = np.sin(x)
  plt.figure(figsize=(8,4))
  plt.plot(x, y, color = 'royalblue', linewidth = 2)
  plt.title(f"{title_text} ({font_name})")
  plt.xlabel("X축 데이터 (시간)")
  plt.ylabel("Y축 데이터 (진폭)")
  plt.grid(True, alpha=0.3) # 격자 (눈금을 정확하게)
  plt.show() # 화면 출력
draw_with_font('NanumGothic', "고딕체 적용 텍스트")
draw_with_font('MyNanumMyeongjo', "명조체 적용 테스트")

# 컬러, 글꼴, ....
print(plt.style.available)

# 비상태 기반
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
def plot_example(style_name):
  plt.style.use(style_name)
  # figure 도화지, ax(axes 영역)
  fig, ax = plt.subplots(figsize=(6,4))
  ax.plot(x, y1, label='Sine', linewidth=2)
  ax.plot(x, y2, label='Cosine', linewidth=2)
  ax.set_title(f"Style: {style_name}")
  ax.legend()
  plt.show()
# 통계/분석, 베이지안(사후 확률), 발표용, 현대적(matplotlib wrapper: 분석결과)
styles = ['ggplot', 'bmh', 'dark_background', 'seaborn-v0_8']
for s in styles:
  plot_example(s)

# ticks 눈금
from numpy.random import randn
# randn의 값의 범위 : 정규분포 sampling -> −∞∼+∞
fig=plt.figure(); # 상태기반, 도화지 : 화면분할, 사이즈
# axes는 영역 - 축 : 눈금
ax=fig.add_subplot(2,2,1) # 행, 열, 순서위치
ax.plot(randn(1000).cumsum()) # 누적합 : 주식 chart simulation
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
ax.set_title('1시간에 따른 누적')
ax.set_xlabel('stages')

ax=fig.add_subplot(2,2,2) # 행, 열, 순서위치
ax.plot(randn(1000).cumsum()) # 누적합 : 주식 chart simulation
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
ax.set_title('2시간에 따른 누적')
ax.set_xlabel('stages')

ax=fig.add_subplot(2,2,3) # 행, 열, 순서위치
ax.plot(randn(1000).cumsum()) # 누적합 : 주식 chart simulation
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
ax.set_title('3시간에 따른 누적')
ax.set_xlabel('stages')

ax=fig.add_subplot(2,2,4) # 행, 열, 순서위치
ax.plot(randn(1000).cumsum()) # 누적합 : 주식 chart simulation
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
ax.set_title('4시간에 따른 누적')
ax.set_xlabel('stages')

plt.show()

# 정적분(면적), 부정적분(원래의 함수)
# 미분 -> 원래의 함수 (부정적분)
# x ** 2 + c -> 2x
# surface : patches
from matplotlib.patches import Polygon # 폐다각형
# R에서 신뢰구간 그릴 때
def func(x):
  return (x - 4) * (x - 6) * (x - 5) + 100
a, b = 2, 9 # 정적분
x = np.linspace(0, 10)
y = func(x)
fig, ax = plt.subplots()
ax.plot(x, y, "k", linewidth = 2)
ax.set_ylim(bottom = 0) # -없음
ix = np.linspace(a, b) # x값
iy = func(ix) # 높이값
# * 분리해서 입력
verts = [(a, 0), *zip(ix, iy), (b, 0)]

poly = Polygon(verts, facecolor = "green", edgecolor = '0.5', alpha = 0.4)
ax.add_patch(poly) # 면추가
ax.text(0.5 * (a + b), 30, r"$\int a ^ b f(x) \mathrm{d}x$",
        horizontalalignment = 'center',
        fontsize = 20)

fig.text(0.9, 0.05, '$x$')
fig.text(0.1, 0.9, '$y$')
# figure -> axes -> axis -> spine (경계선)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks((a,b))
fig.suptitle('정적분\n\n', fontweight="bold")
fig.canvas.draw() # backend layer : canvas : 명시적
plt.show()

# 위상차 시각화 : 90도 (cos : 1에서 시작, sin : 0에서 시작)
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.title('삼각함수 그래프')
plt.plot(X, C, label="cosine")
plt.plot(X, S, label="sine")
plt.xlabel("pi 값 ")
plt.legend(loc = 2)
plt.scatter([0], [0], color="r", linewidth=10)
# 화살표
# latex 문법 (수식, 기호를 출력)
# 화살표 끝지점의 위치
# 데이터가 출력되는 좌표계를 이용해서
plt.annotate(r'$(0,0)$', xy=(0,0), xycoords='data',
             xytext=(-50, 50), textcoords='offset points', fontsize=16,
              arrowprops=dict(arrowstyle="->", linewidth=3, color="g"))
plt.show()

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2)
# cos(회전, 주기) + exp (기하급수적증가감소 : 감쇠 : decay)
# 파동현상 - 저항
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)
ax1 = plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('첫번째 subplots')
print(ax1)
ax2 = plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time(s)')
print(ax2)

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.bar([1,2,3], [3,4,5])
ax2.barh([0.5, 1, 2.5], [0, 1, 2]) # horizontal 수평
ax1.axvline(0.65) # vertical 수직으로
ax2.axhline(0.45) # horizontal 수평으로
plt.tight_layout() # 여백없이
plt.show()

# 축단위의 중요성
# color + shape : red, --. blue square, green, 삼각형
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
# 급격하게 상승 : 축단위 지정에 조심
plt.plot(t, t, 'r--', t**2, t**2, 'bs', t, t**3, 'g^')
plt.show()

import seaborn as sns
sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "NanumGothic",
        "axes.unicode_minus": False
    }
)

points=np.arange(-5, 5, 0.01)
points.size # 1000
# meshgrid, ogrid
# x좌표계, y좌표계

xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs**2 + ys**2)
# imshow : 이미지로 보여줌
plt.imshow(z, cmap=plt.cm.rainbow);
plt.colorbar()
plt.title(" $\sqrt{x^2 + y^2}$")
plt.show()

z = np.sin(xs) * np.cos(ys)

plt.imshow(z, cmap='rainbow')
plt.colorbar()
plt.title("sin(x)cos(y)")
plt.show()

# 양방향 막대 그래프
n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5, 1.0, n)
# 남여, 그룹을 비교할 때 양방향 비교
plt.bar(X, +Y1, facecolor="#9999ff", edgecolor="white")
plt.bar(X, -Y2, facecolor="#ff9999", edgecolor="white")
for x, y in zip(X, Y1):
  plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va='bottom')
plt.ylim(-1.25, +1.25)
plt.show()

# 최소 제곱법
x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([x, np.ones(len(x))]).T
print(A)
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
m, c
# 선형회귀 : R 통계적 검증, python (accuracy)

# 기울기(m)와 절편(c)
plt.plot(x, y, 'o', label='Original', markersize=10)
plt.plot(x, m*x + c, 'r', label="Fitted line")
plt.legend()
plt.show()

# polynomial 다항방정식
t = np.arange(0, 10, 0.01)
y = 2*t + 3
y_noise = y + np.random.randn(len(y))
plt.figure(figsize=(12, 8))
plt.plot(t, y_noise)
plt.show()

# 데이터로부터 계수
# 1차 방정식으로 계수
fp1 = np.polyfit(t, y_noise, 1) # 방정식의 계수, 절편
fp1

f1 = np.poly1d(fp1) # 계수를 이용해서 방정식을 구성
f1

plt.figure(figsize=(12,8))
# 원래 데이터, 노이즈 데이터, 다항방정식으로 구한 해
plt.plot(t, y_noise, label='noise', color='y')
plt.plot(t, y, ls='dashed', lw=3, color='b', label='original')
plt.plot(t, f1(t), lw=2, color='r', label='polyfit')
plt.grid()
plt.legend()
plt.show()

import warnings
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
z = np.polyfit(x, y, 3) # 3차 방정식 계수 추정
z

p = np.poly1d(z)
print(p) # 방정식
print(p(5.0))

# with warnings.catch_warnings():
#   warnings.simplefilter('ignore', np.RankWarning)
p30 = np.poly1d(np.polyfit(x, y, 30)) # 30차 방정식
p30

import matplotlib.pyplot as plt
xp = np.linspace(-2, 6, 100)
_ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2,2)
plt.show()
# 30차는 과적합 (원래 데이터를 다 통과)
# 저차원 일반화된 모델을 구성 - 강건한 모델은 일반화된 모델
# ML/DL 과적합 문제 - 일반화된 모델이 우수한 모델

np.random.seed(42)
x = np.linspace(-3, 3, 40)
true_y = 0.5 * x**3 - 2 * x**2 + x + 3
# rand(0~1), randn(표준 정규분포), normal(정규분포 : 평균, 표준편차)
noise = np.random.normal(0, 3, size=x.shape)
y = true_y + noise
np.save("x.npy", x) # 행과 열이름도 없음 ndarray 순수하게 계산위주
np.save("y.npy", y)
print("x.npy, y.npy 저장 완료")
x = np.load("x.npy")
y = np.load("y.npy")
p0 = np.poly1d(np.polyfit(x, y, 0))
p2 = np.poly1d(np.polyfit(x, y, 2))
p5 = np.poly1d(np.polyfit(x, y, 5))
p50 = np.poly1d(np.polyfit(x, y, 10))
x_plot = np.linspace(x.min(), x.max(), 500)
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(x_plot, p0(x_plot), 'b:', linewidth=2, label='Constant')
plt.plot(x_plot, p2(x_plot), 'b-', linewidth=2, label='Quadratic')
plt.plot(x_plot, p5(x_plot), 'g--', linewidth=2, label='Order 5')
plt.plot(x_plot, p50(x_plot), 'r-.', linewidth=2, label='Order 10')
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title("Polynomial Fitting", fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.show()
# 오차가 너무 심함
# 세부 패턴을 반영하는가
# 과적합은 있는가 : 뒤로 가면 진동이 심해짐

# 문제 - sickts에서는 비선형회귀 다항식으로 해결
import numpy.random as npr
import numpy.linalg as la
x = np.linspace(0,2,11)
y = 6*x*x + 5*x + 2 + npr.normal(0, 0.6, len(x))
plt.plot(x, y, 'o')
# 비선형회귀를 위해서 다항식을 구성 (파생변수)
A = np.vstack([x*x, x, np.ones(len(x))]).T

# 최소제곱법으로 회귀 계수를 구하고 시각화 하시오
b = la.lstsq(A, y)[0]
xi = np.linspace(0,2,11)
yi = b[0]*xi*xi + b[1]*xi + b[2]
plt.plot(xi, yi, 'r-')

xp = np.linspace(0,2,11)
coef = np.polyfit(x, y, 2)
p = np.poly1d(coef)
plt.plot(xp, p(xp), '-')

import numpy as np
scores = np.array([85, 92, 78, 92, 65, 88, 100, 72, 85, 92])
print(f"원본 데이터: {scores}\n")
print("--- [1] 기초 통계량 ---")
print(f"평균 (mean): {np.mean(scores)}")
print(f"중앙값 (median): {np.median(scores)}")
print(f"최대값 (max): {np.max(scores)} (인덱스: {np.argmax(scores)})")
print(f"최소값 (min): {np.min(scores)} (인덱스 : {np.argmin(scores)})")
print(f"점수 범위 (ptp): {np.ptp(scores)}") # max - min
print("--- [2] 분산 및 편차 ---")
print(f"표준편차 (std): {np.std(scores):.2f}")
print(f"분산 (var): {np.var(scores):.2f}")
print("--- [3] 분포 상세 분석 ---")
print(f"1사분위수 (25%): {np.quantile(scores, 0.25)}")
print(f"상위 10% 점수: {np.percentile(scores, 90)}")
print("--- [4] 고유값 및 빈도 분석 ---")
# 범주형(R: table)
val, counts = np.unique(scores, return_counts=True)
print(f"등장한 점수들: {val}")
print(f"각 점수별 인원: {counts}")
study_hours = np.array([8, 10, 7, 11, 5, 9, 12, 6, 8, 10])
# 상관계수행렬을 리턴 (정방행렬, 대칭행렬)
# 2x2
corr = np.corrcoef(study_hours, scores)[0, 1]
print(f"공부시간과 성적 간의 상관계수: {corr:.4f}")

data = np.array([1, 2, 2, 3, 4, 5, 5, 5, 6])
print("히스토그램:", np.histogram(data, bins=5))

# 행렬인 경우
# 없으면 : 전체
# 0 : 행방향 즉 열로 계산
# 1 : 열방향 즉 행으로 계산
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
# 1,  2,  3
# 4,  5,  6
# 7,  8,  9
print("원본\n", arr)
print("누적합(행)=\n", arr.cumsum(0)) # cumulative 누적
print("누적곱(열)=\n", arr.cumprod(1)) # 곱하기 누적
arr=np.random.randn(5,4)
print(arr)
print("배열의 평균=", arr.mean()) # 전체 평균
print(np.mean(arr))
print("배열의 합계=", arr.sum()) # 전체 합계
print("열방향으로의 평균=", arr.mean(axis=1)) # 행 평균
print("행방향으로의 합계=", arr.sum(0)) # 열합계

vec1 = np.arange(5)
vec2 = np.arange(1,6)
print(vec1)
print(vec2)
print(np.sum(vec1 * vec2))

print(np.sqrt(np.sum(vec1**2))) # 피타고라스
np.linalg.norm(vec1)

# 내적
costheta = (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 길이의 비 -> 라디안
np.arccos(costheta)
# 라디안 -> 각도
np.degrees(np.arccos(costheta))

# cosine similarity 코사인 유사도 함수
# -1 ~ 1
def cos_sim(vec_1, vec_2):
  return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * (np.linalg.norm(vec_2)))

A = np.array([[1,2],[3,4]]) # 2x2
print(A)
A ** 2 # 요소 제곱

# 행렬 거듭제곱 (변환의 연속 누적)
# 행렬 MRS(move, rotate, scale)
A @ A # 거듭제곱은 정방행렬만 가능

A.T @ A

A*A # 요소곱

arr=np.arange(32).reshape((8,4))
print(arr)

arr.T @ arr # 열간 관계 분석

arr @ arr.T # 행과 행 사이의 관계

np.matmul(arr, arr.T) # matrix multiply

# tile
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
print(x)
v = np.array([1, 0, 1])
print("타일출력")
vv = np.tile(v, (4, 1)) # 반복 배치 : 행으로 4번 반복
print(vv)

y = x + vv
print(y)

x+v # broadcasting

# 면, 행, 열
# 0, 1,  2
arr = np.arange(24).reshape((2,3,4))
print(arr)

print(arr.transpose((1,0,2))) # 면과 행을 전치
                    # 3x2x4
print(arr.transpose((0,2,1))) # 행과 열을 전치
                    # 2x4x3

# 과제

# x + y + 0*z = 2
# 0*x + y + z = 2
# x + y + z = 3 방정식의 해를 구하시오
A = np.array([[1,1,0], [0,1,1], [1,1,1]])
B = np.array([2,2,3])
x = np.linalg.solve(A, B)
print(x)

# 문제 : 다음 두 벡터의 사이각을 구하시오
a = np.array([0,1,0])
b = np.array([1,0,0])
dot = a @ b
print(dot)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print(norm_a, norm_b)
cos_theta = dot / (norm_a * norm_b)
print(cos_theta)
theta = np.arccos(cos_theta)
print(theta)
degree = np.degrees(theta)
print(degree)

# 문제 : x = np.array([[1,2], [3,4]])
# 이 행렬에 대하여 역행렬을 구하고 행렬곱을 해서 결과가 단위 행렬임을 확인하시오
x = np.array([[1,2], [3,4]])
x2 = np.linalg.inv(x) # inverse : solve로 구할 수 있음
print(x2)
print(x @ x2)

# I = np.eye(2)
# x_inv = np.linalg.solve(x, I)
# print(x_inv)

# 문제3
# 삼각함수 than(theta) = sin(theta) / cos(theta) 임을 검증하시오
# 삼각함수 cos(theta)**2 + sin(theta)**2 = 1임을 검증하시오
theta = np.pi / 4
print(theta)

than = np.tan(theta)
ans = np.sin(theta) / np.cos(theta)
print(than)
print(ans)

res = np.cos(theta)**2 + np.sin(theta)**2
print(res)