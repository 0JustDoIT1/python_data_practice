from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open("/content/drive/MyDrive/dataset/gom.png")
plt.imshow(img)
print(type(img))  # PIL.PngImagePlugin.PngImageFile
plt.show()


print(np.array(img).shape)
nimg = np.array(img)  # 이미지 -> ndarray
print(nimg.shape)
nimg.ndim

nimg[5, :]

# 부동소수점 이미지
# [255, 255, 255] -> 그래픽 카드 -> 1pixel
# 이미지 정규화가 필요한 때 : deep learning : 거리값등에서
image_data = np.array(img).astype(float) / 255
print(image_data.dtype) # float64
plt.imshow(image_data)
plt.show()

image_data[:3, :]

# PIL -> imageio -> openCV (object detection)
import imageio
face_read = imageio.imread('/content/drive/MyDrive/dataset/gom.png')
plt.imshow(face_read)
plt.show()

from scipy import ndimage, datasets
face = datasets.face()
plt.imshow(face, vmin = 0, vmax = 255) # 컬러값 0~255 다른값을 제한
plt.show()

# 회전
from scipy import ndimage
# 행렬곱을 편하게 함수로 만듦
# [cos(30)], -sin(30)]
# [sin(30),   cos(30)]
rotated_face = ndimage.rotate(face, 30) # defalut 반시계방향
plt.imshow(rotated_face, cmap = plt.cm.gray)
plt.axis('off')

face.shape

# 일부만 크롭
cropped_face = face[100:-100, 100:-100]
plt.imshow(cropped_face, cmap = plt.cm.gray)
plt.axis('off')

# 3차원 이미지 축 0, 1, 2 / 2는 컬러값
# (255 + 255 + 255) 컬러값의 범위를 넘어섬
face2 = face.sum(axis = 2)/ 3 # 컬러값을 1개로 만드는 방법
print(face2.shape)  # (768, 1024, 3) -> (768, 1024) 하나의 컬러값(흑백)일 때 1은 생략 (769, 1024, 1)
zoomed_face = ndimage.zoom(face2, 0.5)
print(zoomed_face.shape) # (384, 512)
plt.imshow(zoomed_face, vmin = 0, vmax = 255)
plt.axis('off')

face6 = face.astype(float)/255
print(face6.shape) # (768, 1024, 3)의 부동소수점 이미지
lx, ly, lz = face.shape
# 이미지 사이즈 줄어듦
face6 = face6[int(lx / 4): -int(lx / 4), # 이미지를 1/4 줄인 것
             int(ly / 4): -int(ly / 4)]
             # (768, 1024, 3) * [1.0, 0.5, 0.5]
             # broadcasting  768*1024 횟수만큼 곱셈
img_tinted = face6 * [1.0, 0.5, 0.5] # RGB 컬러의 강도 변화
plt.subplot(1, 2, 1)
plt.imshow(face)
plt.subplot(1, 2, 2)
plt.imshow(img_tinted)
plt.show()
print(img_tinted.shape)

# unit 8: unsigned 1byte 0~255
# int8 : -128 ~ 127까지
print(f'데이터 타입: {face.dtype}')
print(f'1개 요소의 바이트 크기: {face.itemsize} bytes')
print(f'전체 이미지의 바이트 크기: {face.nbytes} bytes')

print(face.shape)
lx, ly, lz = face.shape
byte_calc = lx * ly * lz
print("바이트:", byte_calc)
# 바이트를 킬로 바이트로 변환하시오
print("killo바이트 :", byte_calc / 1024)

# 등고선 표현
face = datasets.face(gray = True)
plt.contour(face, [10, 100, 200]) # 등고선
# 이미지가 출력될 때는 좌상단이 0,0 거꾸로 자람

# 똑바로
face = datasets.face(gray = True)
plt.contour(face, [10, 100, 200])
plt.gca().invert_yaxis()

face = datasets.face(gray = True)
print(face.shape)
lx, ly = face.shape  # 이미지크기 / 2 -> 중심
X, Y = np.ogrid[0:lx, 0:ly]
print(X.shape)
print(Y.shape)
# 브로드캐스팅

# 중심 +,-값
# 피타고라스 정리
# 거리값은 중심으로부터 원형으로 같은 위치가 표현
mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
print(mask)
face[mask] = 0  # 검은색

mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 < lx * ly / 32
face[mask] = 0
plt.imshow(face)

face = datasets.face(gray = True)
mask1 = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 < lx * ly / 4
mask2 = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 8
mask3 = mask1 & mask2
face[mask3] = 200
plt.imshow(face)

from skimage.color import rgb2gray
from skimage.io import imread
from scipy import ndimage as ndi
img = rgb2gray(imread(("/content/drive/MyDrive/dataset/곰.png")))
w, h = img.shape
plt.imshow(img)
print(w, h)

# 아핀변환과 동차좌표계
# affine 변환 : MRS 변환(move, rotate, scale)
# 동차좌표계: 4 x 4
# 3차원 -> cos theta 0 -sin(theta) 확장
#           0        0     0       확장
#         sin(theta) 0 cos(theta)  확장
#            movex  movey  movez   확장

mat_identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
img1 = ndi.affine_transform(img, mat_identity)
plt.imshow(img1)
plt.show()

# 좌우 반전
# [1, 0, 0], x, y, z -> y축 y요소를 -
# [0, -1, 0],
# [0, 0, 1]
# 행렬곱 -> 변환의 합성 : 행렬곱 -> 변환을 여러번
mat_reflect = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, 1, -h], [0, 0, 1]])
img1 = ndimage.affine_transform(img, mat_reflect)
plt.imshow(img1)
plt.show()

theta = np.pi/4  # 45도

# 이동행렬 * 회전행렬 * 역이동행렬
# x축, y축, 이동
# 회전한다면 원점을 중심하고 회전
# 중간을 원점으로 해서 회전시키기 위해
# 동차좌표계 : MRS를 다 표현하기 위해서 1차원 늘려
mat_rotate = np.array([[1, 0, w/2], [0, 1, h/2], [0, 0, 1]]) @ np.array([[np.cos(theta), np.sin(theta), 0],
                                                                         [np.sin(theta), -np.cos(theta), 0], [0, 0, 1]])  @ np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])

mat_rotate

img1 = ndi.affine_transform(img, mat_rotate)
plt.imshow(img1)
plt.show()

from scipy import linalg
import numpy as np
A = np.array([[1, 2], [3, 4]])
A = A @ A.T # 자기행렬 행렬곱을 하면 정방행렬, 대칭행렬이 생성
l, v = linalg.eig(A) # 고유값 분해
print(l) # 고유치 (축 방향으로의 분산)
print(v) # 고유벡터 (벡터 간의 내적은 0 : 직교)

# 역행렬을 만들 수 없으면 에러가 발생
from scipy.linalg import *
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([1, 2, 3])
# x = solve(A, b)

det = linalg.det(A) # determinent가 0이면 역행렬이 안됨
print(f"계산된 행렬식(Det) : {det}")

x, residuals, rank, s = linalg.lstsq(A, b)
print("근사해 x:", x)

# 문제
# 2x + y + z = 1
# 4x + 3y + 4z = 2
# -4x + 2y + 2z = -6
# 이 방정식의 해를 구하시오
A = np.array([[2, 1, 1], [4, 3, 4], [-4, 2, 2]])
b = np.array([1, 2, -6])
det = linalg.det(A)
print(det)
# x, residuals, rank, s = linalg.lstsq(A, b)
# print("해:", x)

threshold = 1e-9
if abs(det) < threshold:
  print("특이 행렬이 감지되어 최소자승법(lstsq)으로 근사해")
  x, residuals, rank, s = linalg.lstsq(A, b)
  print("근사해 x:", x)
else:
  x = linalg.solve(A, b)
  print("정확한 해 x:", x)

# 특이행렬분해
A = np.array([[1, 2, 3], [4 ,5, 6]]) # 2 x 3
A

U, s, Vh = linalg.svd(A) # 특이행렬분해
U, s, Vh # 행특징 * s는 고유치(1/2) * 열특징

Sig = linalg.diagsvd(s, 2, 3)
Sig

U.dot(Sig.dot(Vh))

A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2, 3])
# 의사역행렬 (가짜 역행렬)
A_pinv = np.linalg.pinv(A)
# 근사해를 구함
x_pinv = np.dot(A_pinv, b)

# 밀도함수
# scipy
from scipy.stats import norm # 표준 정규분포
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
# 평균, 분산, 왜도, 첨도 # 0.0 1.0 0.0 0.0
# 첨도의 기준은 3 초과 첨도
# 0을 기준(-3을 뺀 상태에서 판단)
mean, var, skew, kurt = norm.stats(moments = 'mvsk')
print(mean, var, skew, kurt)
# dnorm
# ppf 분위수(확률값)
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
# pdf 밀도를 구하는 함수
ax.plot(x, norm.pdf(x), 'r-', lw = 5, alpha = 0.6, label = 'norm pdf')
rv = norm() # 인스턴스 해서 시각화
ax.plot(x, rv.pdf(x), 'k-', lw = 2, label = 'frozen pdf')

fig, ax = plt.subplots(1, 1)
r = norm.rvs(size = 100000) # rnorm
ax.hist(r, density = True, histtype = 'stepfilled', alpha = 0.2)
ax.legend(loc = 'best', frameon = False)
plt.show()

# 문제
# 분위수 -1에서 1사이의 확률값을 출력하시오 (cdf)
# 95% 신뢰구간일 때 분위수 값을 출력하시오 (ppf)
from scipy import stats
print(stats.norm.cdf(1) - stats.norm.cdf(-1))
stats.norm.ppf([0.025, 0.975])

# 1 sample t-test
Y = stats.norm()
print(len(Y.rvs(size = 1000)))
stats.ttest_1samp(Y.rvs(size = 1000), 0.1)
# 0.0085
# 귀무가설을 기각하고 대립가설 채택 : Y는 모집단과 다르다

data1 = Y.rvs(size = 1000)
data2 = Y.rvs(size = 1000)
t_statistics, p_value = stats.ttest_ind(data1, data2)
print("t 통계값", round(t_statistics, 2))
print("p_value", round(p_value, 2))
# 귀무가설을 기각할 수 없다.

dalc = np.array([6.47,6.13,6.19,4.89,5.63,4.52,5.89,4.79,5.27,6.08])
dtob = np.array([4.03,3.76,3.77,3.34,3.47,2.92,3.2,2.71,3.53,4.51])
stats.pearsonr(dalc, dtob)

# 선형회귀
print(stats.linregress(dalc, dtob)) # dalc : 독립변수 dtob : 종속변수
# 방법 2
result = stats.linregress(dalc, dtob)
result.slope

result.intercept

# 독립변수와 종속변수간의 상관계수
mean_x = np.mean(dalc)
mean_y = np.mean(dtob)
mean_x

# 독립변수의 평균으로부터의 차 * 종속변수의 평균으로부터의 차
numerator_r = np.sum(dalc - mean_x) * (dtob - mean_y) # 분산
numerator_r

ss_x = np.sum((dalc - mean_x) ** 2) # 분산
ss_y = np.sum((dtob - mean_y) ** 2)

slope = numerator_r / ss_x # 독립변수의 분산 : 기울기
intercept = mean_y - slope * mean_x # 절편은 평균으로 계산
intercept

# 설명력 = 회귀제곱합 / 전체제곱합
r_value = numerator_r / np.sqrt(ss_x * ss_y) # 설명력

# 다항 방정식

# 방정식이 다항 방정식으로 고정되어 있음
from scipy.optimize import curve_fit
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
# 계수를 확인
# y = ax^3 + bx^2 + cx + d
# y = 0.08703704 * x**3 -0.81349206 * x**2 + 1.69312169 * x -0.03968254
# 오차 = 예측값 - 실제값
z = np.polyfit(x, y, 3) # 3차 식으로 적합 - 계수찾기
print("3차 다항식의 계수", z)
p = np.poly1d(z) # 계수들을 이용해서 수식 작성 - 방정식 작성
print(p)
print(p(0.5)) # 0.614384920634922
p(3.0) # np.float64(0.06825396825396943)
p(10)

# 시각화
xp = np.linspace(-2, 6, 100)
p30 = np.poly1d(np.polyfit(x, y, 30))
# 3차 방정식, 30차 방정식 비교
_ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2, 2)
plt.show()

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
plt.rcParams.update({'font.size': 14, 'font.family': 'NanumGothic', 'axes.unicode_minus': False})

# curve_fit는 목적함수를 지정 가능
def func(x, a, b, c, d): # 4개의 계수를 추정
  return a * np.sin(b * x + c) + d # 진폭
# 초기값으로 시작점을 지정
# optimization : 여러번 계산 ( 가장 최적의 값을 리턴)
# covariance : 계수들 간의 공분산
params, cov = curve_fit(func, x, y, p0 = [1.0, 1.0, 0.0, 0.0])
print(params, cov) # 공분산 : 앞뒤를 전치해도 가능
# 대각선은 자기의 분산
#           a                b               c               d
# a [ 1.93349446e-04  1.08885925e-05 -3.24211345e-05  5.63184980e-07]
# b [ 1.08885925e-05  1.16503439e-04 -2.18415738e-04  7.84584917e-05]
# c [-3.24211345e-05 -2.18415738e-04  6.06096493e-04 -1.37968949e-04]
# d [ 5.63184980e-07  7.84584917e-05 -1.37968949e-04  1.50809677e-04]]
a_opt, b_opt, c_opt, d_opt = params
plt.figure(figsize = (8, 6))
plt.scatter(x, y, label = 'Original Data')
x_fit = np.linspace(0, 5, 100)
y_fit = func(x_fit, a_opt, b_opt, c_opt, d_opt)
plt.plot(x_fit, y_fit, 'r-', label = 'Fitted Curve')
plt.title('사인함수로 적합')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

from scipy import optimize
def f(x):
  return 4*x**3 + (x-2)**2 + x**4

fig, ax = plt.subplots()
x = np.linspace(-5, 3, 100)
ax.plot(x, f(x))

# 뉴톤법은 지역해 염려가 있음
x_min = optimize.fmin_bfgs(f, 1) # 1근방에서 시작점
x_min # 0.46 - 지역해

optimize.brent(f) # 무작위 대입법

optimize.fminbound(f, -4, 2) # np.float64(-2.6729822917513886)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # 3차원 출력
def fm(p): # 높이값을 결정
  x, y = p
  return (np.sin(x) + 0.05 * x ** 2
          + np.sin(y) + 0.05 * y ** 2)
x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y) # 그물망(2500개)
Z = fm((X, Y))

fig = plt.figure(figsize = (9, 6))
ax = fig.add_subplot(projection = '3d')
# 면 : wireframe 선
# rowstride, columnstride 건너뛰기
surf = ax.plot_surface(X, Y, Z, rstride = 2, cstride = 2, cmap = mpl.cm.coolwarm, linewidth = 0.5, antialiased = True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink = 0.5, aspect = 5)

# 전역해를 구하는 함수
import scipy.optimize as spo
# 지역해 : 뉴톤법(지역해에 빠질 경우 많음) - 정밀하게 찾아야 할 때
# 전역해 : brute - 무작위
# 곡선피팅 : curve_fit(목적함수로부터)
# 방정식의 해 : Root
output = True
def fo(p):
  x, y = p
  z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
  if output == True:
    print('%8.4f %8.4f %8.4f' % (x, y, z))
  return z
# 격자형으로 무작위하게 탐색(최저점)
# finish : 격자형으로 찾은 다음 정밀하게 최종값을 구할 때는 지역해를 도입
opt0 = spo.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish = None) # [0. 0.], [-1.4 -1.4]
print(opt0)

# 기호수학
import sympy as sp
x = sp.Symbol('x')
y, z = sp.symbols('y z', real = True)
expr = x + y*z
print(expr)

x = sp.Symbol('x') # 변수
f = sp.sin(x) * x**2 # 함수 지정
df = sp.diff(f, x) # 미분
print(df)
# x**2 미분 => 2x


x_vals = np.linspace(-5, 5, 400)
# lambdify : 기호를 numpy 수식으로 변경하는 것
f_vals = sp.lambdify(x, f)(x_vals) # 사인함수 f = sp.sin(x) * x**2
print(sp.lambdify(x, f)) # <function _lambdifygenerated at 0x7cc36648fb00>
print(f_vals)
df_vals = sp.lambdify(x, df)(x_vals)
plt.figure(figsize = (10, 6))
plt.plot(x_vals, f_vals, label = "f(x) = sin(x) * x^2", color = 'blue')
plt.plot(x_vals, df_vals, label = "df(x) = 2 * x * sin(x) + x^2 * cos(x)", color = 'red', linestyle = '--')
plt.xlabel("x")
plt.ylabel("y")
plt.title("함수와 미분함수")
plt.xlim(-5, 5)
plt.ylim(-10, 10)
plt.grid(True)
plt.legend()
plt.show()

# 과제
import pandas as pd
import matplotlib.pyplot as plt
read_file = pd.read_csv('/content/drive/MyDrive/dataset/play_top30.csv', skiprows = 1)
read_file.head()
# 랭킹, 플레이어이름, 팀이름, 포지션, 게임수, 골수, 어시스트
# 포인트 득점과 실점, 팀이 승리하고 있을 때(ppp), SHG(팀이 지고 있을 때 ) SHP(팀이 지고 있을 때의 포인트), GWG(골든골)
# OT 연장전에서 넣은 골수, S:슛한 횟수, S% 골 성공률

a = read_file.describe()
a.boxplot()
plt.show()

re_file = read_file.rename(columns = {'P': 'points', 'G': 'goals', 'A': 'Assists', 'S%': 'shooting_percentage', 'Shift/GP': 'shifts_per_game_players'})

re_file.columns

print(re_file.head())

print(re_file.columns)

# 문제1 : goals, Assists, points 간의 상관계수를 출력하시요
re_file[['goals', 'Assists', 'points']].corr()
# 문제2 : # 각 열중에  ["player","points","goals","assists] 열만 출력하는 함수를 작성하시요
# 문제3 : 각 열 중  lower < points <= upper 인 데이터 중 [player, points, goals, assists] 열만 출력하는 함수를 작성하시요
# 문제4:  goals와 assists 간에 선형회귀를 구현하시요
# 문제5 : goals가 [50,70,40,20] 인 경우의 assists를 predict하시요