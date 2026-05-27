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

from google.colab import drive
drive.mount('/content/drive')

from scipy import *
import scipy.constants as sc
print(sc.g)
print(sc.e) # 자연대수
print(sc.pi)

from scipy.constants import inch
meter = 10 * inch
print(meter)

import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist) # 100*100 => 10,000
Z = np.sqrt(X**2 + Y**2) # 거리값 - 피타고라스 정리
fig, ax = plt.subplots(1,1)
# 등고선 : fill => 출력 2차원
# gradation
# histogram bin 구간
cp = ax.contourf(X, Y, Z, levels = 6, cmap='winter')
fig.colorbar(cp)
ax.set_title('Filled Contours Plot')
ax.set_ylabel('y (cm)')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def fm(p): # x,y 변수 : 조합
  x, y = p
  return np.sin(x) + 0.25 * x + np.sqrt(y)
x = np.linspace(0, 10, 20) # 제약사항
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)
Z = fm((X, Y))
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(projection = '3d') # 문법이 변경됨
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                       cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Function Surface with Minimum Point')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()
plt.show()
# 다음 optimization : 최저점

from scipy.optimize import minimize # 최적화 함수
# 뉴톤법은 초기값이 필요
initial_guess = [2.0, 2.0] # 초기값 - 최적화 시작점
spatial_bounds = ((0, 10), (0, 10)) # 탐색 범위
opt_result = minimize(
    fun=fm,
    x0 = initial_guess, # 초기값
    method='L-BFGS-B', # 알고리즘 : quasi-newton 법 (경사하강법 + hessain 곡률근사)
    bounds=spatial_bounds # 탐색 범위
)
best_x, best_y = opt_result.x
best_z = opt_result.fun # 최적의 함수값
print("=== 최적화 탐색 결과 보고서 ===")
print(f"최적화 성공 여부 : {opt_result.success}")
print(f"찾아낸 최적의 x 위치: {best_x: .4f}")
print(f"찾아낸 최적의 y 위치: {best_y: .4f} (루트 함수 특성상 0으로 수렴)")
print(f"지형의 최소값 f(x,y): {best_z: .4f}")

# 문제 : 위 시각화된 부분에 최소값을 시각화해서 추가하시오
import matplotlib as mpl
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(projection = '3d') # 문법이 변경됨
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                       cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
# zorder : layer 순서
ax.scatter(best_x, best_y, best_z, color='red', s=150, marker='*', zorder=20, label="Minimum Point")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Function Surface with Minimum Point')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': '3d'})
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2) # 거리값
Z = R # (0, 0) 점이 최저점
Z = np.sin(R) # 주기값
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
plt.show()
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.rainbow)
plt.show()

# 문제
def f(x, y): return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)
fig, ax = plt.subplots(figsize = (9,6), subplot_kw={'projection': '3d'})
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(f(X, Y))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
ax.view_init(elev=15, azim=45) # 카메라 위치조절
plt.show()

# 위 도면을 이용해서 등고선 맵을 출력하시오
# 경계선도 출력하시오
# colorbar를 이용해서 높이값을 확인하시오
surf = plt.contourf(X, Y, Z, alpha=0.75, cmap='jet')
line = plt.contour(X, Y, Z, colors='black', linewidths=0.5)
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

# 보간법(interpolation)
# 없는 데이터를 추가 : 시각화, 결측치 처리
from scipy.interpolate import *
def f(x):
  return np.sin(x)
n = np.arange(0, 10)
x = np.linspace(0, 9, 100)
y_meas = f(n) + 0.1 * np.random.randn(len(n)) # 노이즈 추가(측정)
y_real = f(x) # 원래데이터
# 기본 : kind='linear' 선형
# 구간함수
linear_interpolation = interp1d(n, y_meas, kind='linear')
y_interp1 = linear_interpolation(x)
cubic_interpolation = interp1d(n, y_meas, kind='cubic')
y_interp2 = cubic_interpolation(x)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(n, y_meas, 'bs', label='noisy data')
ax.plot(x, y_real, 'k', label='true function')
ax.plot(x, y_interp1, 'r', label='linear interp')
ax.plot(x, y_interp2, 'g', label='cubic interp')
ax.legend(loc=3)

# 미분
def f(x): # original
  return x**3 - 3 * x**2 + x
def fprime(x): # 1차 미분
  return 3 * x**2 - 6 * x + 1
def fprime2(x): # 2차 미분
  return 6 * x - 6

x = np.linspace(-1, 3, 400)
plt.figure(figsize=(10, 15))
plt.subplot(311)
plt.plot(x, f(x))
plt.xlim(-2, 4)
plt.xticks(np.arange(-1, 4))
plt.yticks(np.arange(-5, 4))
plt.title('f(x)')
plt.subplot(312)
plt.plot(x, fprime(x))
plt.xlim(-2, 4)
plt.xticks(np.arange(-1, 4))
plt.yticks(np.arange(-3, 11))
plt.title("f'(x)")
plt.subplot(313)
plt.plot(x, fprime2(x))
plt.xlim(-2, 4)
plt.xticks(np.arange(-1, 4))
plt.show()

# 편미분 / 상미분
import sympy

x, y = sympy.symbols('x y') # 변수 지정
f = x**2 + x * y + y**2
f

# 동시미분이 불가능 -> x입장에서 미분, y입장에서 미분
# 변수가 2개 -> 편미분을 2개 -> 결합을 해서 변화량이 많은 곳으로 이동
sympy.diff(f, x) # x변수를 중심으로 편미분, y 변수는 상수 취급
# [2x + y, x + 2y] # 경사하강법의 대상

sympy.diff(f, y) # y변수를 중심으로 편미분, x는 상수

# 정규분포
x, mu, sigma = sympy.symbols('x mu sigma')
f = sympy.exp((x - mu) **2 / sigma ** 2)
f

sympy.simplify(sympy.diff(f, x)) # 복잡한 함수 - 단순하게 변환
# 지수함수는 그 자체가 미분함수
# 삼각함수 -> 지수함수 (오일러 정리)

x, y = sympy.symbols('x y')
f = 2 * x + y
f # 미분된 식

# 부정적분 (미분 이전의 수식으로 복원)
sympy.integrate(f, x) # 적분

# 정적분, 부정적분
import scipy.integrate as sci
import numpy as np
def f(x):
  return np.sin(x) + 0.5 * x
a = 0.5
b = 9.5
x = np.linspace(0, 10)
y = f(x)
from matplotlib.patches import Polygon
fig, ax = plt.subplots(figsize=(7,5))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)
lx = np.linspace(a, b)
ly = f(lx)
verts = [(a, 0)] + list(zip(lx, ly)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)
plt.text(0.75 * (a + b), 1.5, r"$Wint_a^b f(x)dx$", horizontalalignment='center', fontsize=20)

# 정적분 함수 quad
# 표준정규분포 -1 ~ 1 사이는 68%
# 표준정규분포의 mu = 0, 표준편차(sigma) = 1
from scipy.constants import pi
def normal_pdf(x):
  normalization_factor = 1.0 / np.sqrt(2.0 * pi)
  return normalization_factor * np.exp(- (x ** 2) / 2.0)
lower_bound = -1.0
upper_bound = 1.0
# 정적분 -> 면적을 구함
integral_result, estimated_error = sci.quad(normal_pdf, lower_bound, upper_bound)
print(f"1 표준편차 범위(-1 ~ 1)의 면적 (확률) : {integral_result: .6f}")
print(f"적분 계산 추정 오차 (Precision Error): {estimated_error: .6e}")

from scipy import stats
cdf_result = stats.norm.cdf(1) - stats.norm.cdf(-1)
print(f"\n[비교] stats.norm.cdf를 이용한 계산값 : {cdf_result: .6f}")
print(f"두 방식의 차이 (오차): {abs(integral_result - cdf_result): .6e}")

# 몬테 칼로 (확률식) -> 적분 (면적)
# 표준정규분포 밀도 함수를 이용
np.random.seed(42)
num_samples = 50000 # 많으면 정확
x_min, x_max = -1.0, 1.0 # 분위수
y_min, y_max = 0.0, normal_pdf(0.0) # 밀도 제한
# 50000개의 좌표 조합
x_samples = np.random.uniform(x_min, x_max, num_samples)
y_samples = np.random.uniform(y_min, y_max, num_samples)
is_inside = y_samples <= normal_pdf(x_samples) # x값 밀도 : 파란색
# 면적
box_area = (x_max - x_min) * (y_max - y_min)
# random 확률
mc_integral = box_area * (np.sum(is_inside) / num_samples)
print("=== 몬테카를로 적분 결과 ===")
print(f"총 샘플 개수: {num_samples} 개")
print(f"곡선 아래에 명중한 샘플 수: {np.sum(is_inside)} 개")
print(f"몬테카를로 근사 면적: {mc_integral: .6f}")
print(f"실제 수학적 정답 수치: 0.682689")
# 정확한 값을 구하는 것이 아니라 복잡한 문제를 해결하기 위해서 근사값을 찾음
# 컴퓨터 수치해석 자체가 근사값을 구하는 것
# 비선형이기 때문에

# 함수가 너무 고차원
# 딥러닝 (가중치의 열이 함수식 100)
plt.figure(figsize=(10, 6))
visual_idx = np.random.choice(num_samples, 2000, replace=False)
plt.scatter(x_samples[visual_idx][is_inside[visual_idx]],
            y_samples[visual_idx][is_inside[visual_idx]],
            color='blue', s=3, alpha=0.5, label='Accepted (Inside)')
plt.scatter(x_samples[visual_idx][~is_inside[visual_idx]],
            y_samples[visual_idx][~is_inside[visual_idx]],
            color='red', s=3, alpha=0.3, label='Rejected (Outside)')
x_axis = np.linspace(-2, 2, 200)
plt.plot(x_axis, normal_pdf(x_axis), 'k-', lw=2, label='True PDF')
plt.axvline(x=-1, color='gray', linestyle='--')
plt.axvline(x=1, color='gray', linestyle='--')

plt.title(f"Monte Carlo Integration (Samples: {num_samples}, Result: {mc_integral: .4f})")
plt.xlabel('X')
plt.ylabel('Density')
plt.xlim(-1.5, 1.5)
plt.ylim(0, 0.45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 상미분과 푸리에 변환
# 상미분 : 변수가 하나인 미분
# 1번 미분 : 속도
# 2번 미분 : 가속도

# 감쇠 진동
def dy(y, t, zeta, w0):
  x, p = y[0], y[1]
  dx = p # 속도
  dp = -2 * zeta * w0 * p - w0**2 * x # 가속도
  return [dx, dp]

y0 = [1.0, 0.0] # 초기값
t = np.linspace(0, 10, 1000)
w0 = 2 * np.pi * 1.0 # 진동수

from scipy.integrate import odeint, ode
y1 = odeint(dy, y0, t, args=(0.0, w0))
y2 = odeint(dy, y0, t, args=(0.2, w0)) # 감쇠값
y3 = odeint(dy, y0, t, args=(1.0, w0))
y4 = odeint(dy, y0, t, args=(5.0, w0))

fig, ax = plt.subplots()
# 진동 - 주기적
ax.plot(t, y1[:,0], 'k', label="undamped", linewidth=0.25)
# 감쇠를 너무 약하게 주어서 여진이 있는 모습
ax.plot(t, y2[:,0], 'r', label="under damped")
# 임계값 (급하게 감쇠)
ax.plot(t, y3[:,0], 'b', label="critical damping")
# 진동은 없는 느슨한 감쇠
ax.plot(t, y4[:,0], 'g', label="over damped")
ax.legend()

from numpy.fft import fftfreq, fft
from scipy.fftpack import *

N = len(t)
dt =t[1] - t[0]
F = fft(y2[:,0]) # 푸리에 변환 : 주파수를 확인
w = fftfreq(N, dt) # 그래프로 표현하기 위해 시차값을 x축 계산
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(w, abs(F))
# 음의 주파수, 양의 주파수

# 양의 주파수를 이용해서 Hz를 분석
# 구성 주파수를 확인 감쇠의 범위 확인
# 1H 주파수를 사용, 감쇠 때문이다
indices = np.where(w > 0)
w_pos = w[indices]
F_pos = F[indices]
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(w_pos, abs(F_pos))
ax.set_xlim(0, 5)

# 경사하강법
import scipy.optimize as spo
def cost_function(p):
  x, y = p
  return (x - 3)**2 + (y + 2)**2

# 비선형 -> 컴퓨터 (찾아야 함)
def gradient_function(p): # 1차 편미분
  x, y = p
  df_dx = 2 * (x - 3)
  df_dy = 2 * (y + 2)
  return np.array([df_dx, df_dy])
initial_guess = [0.0, 0.0]
result = spo.minimize(
    fun=cost_function, # 비용, 목적함수
    x0=initial_guess, # 초기값
    method='CG', # conjugate(직교성질) gradient (켤레 경사하강법)
    jac=gradient_function, # 미분값
    options={'disp': True}
)
print(f"최적화 성공 여부: {result.success}")
print(f"찾아낸 전역 최적점 (x, y): {result.x}")
print(f"최적점에서의 함수 최소값 (z): {result.fun}")
# 경사하강법 : 기울기 -> zigzag로 이동
# 뉴톤법 : 기울기 + 곡률 (정교, 시간이 오래 걸림, 지역해에 빠질 수 있음, 발산 위험 있음)
# quaci-newton법 (기울기 + 해시안(2차 미분) 근사)

from scipy import ndimage, datasets
face = datasets.face()
plt.imshow(face, vmin=0, vmax=255)
plt.show

from PIL import Image
img = Image.fromarray(face)
gray_img = img.convert('L')
gray_np = np.array(gray_img)

# 표준편차 3배수 주변의 픽셀의 영향을 고려해서 평균값으로 변환
blurred_face = ndimage.gaussian_filter(gray_np, sigma=3)
plt.imshow(blurred_face)
plt.show()

# Feature extraction
# edge detection
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im = ndimage.rotate(im, 15, mode='constant')
im = ndimage.gaussian_filter(im, 8)
sx = ndimage.sobel(im, axis=0, mode='constant')
sy = ndimage.sobel(im, axis=0, mode='constant')
sob = np.hypot(sx, sy) # 두 개를 피타고라스 정리로 묶어주는
plt.figure(figsize=(16, 5))
plt.subplot(141)
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title('square', fontsize=20)
plt.subplot(142)
plt.imshow(sx)
plt.axis('off')
plt.title('Sobel (x direction)', fontsize=20)
plt.subplot(143)
plt.imshow(sob)
plt.axis('off')
plt.title('Sobel filter', fontsize=20)

np.random.seed(1)
n = 10
l = 256
im = np.zeros((l, l))
points = l * np.random.random((2, n**2))
im[(points[0]).astype(int), (points[1]).astype(int)] = 1
# 노이즈를 제거효과 gaussian_filter
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
mask = (im > im.mean()).astype(float)
mask += 0.1 * im
img = mask + 0.2 * np.random.randn(*mask.shape)
# 연속적 데이터를 bin 구간으로 분리하고 카운트
# 이미지를 배경과 전경 : 밝은 색 vs 어두운 색
hist, bin_edges = np.histogram(img, bins=60) # 어디를 기준으로
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # 중앙점
binary_img = img > 0.5 # true / false
plt.figure(figsize=(11, 4))
plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)
plt.plot(bin_centers, hist, lw=2)
plt.axvline(0.5, color='r', ls='--', lw=2)
plt.text(0.57, 0.8, 'histogram', fontsize=20, transform=plt.gca().transAxes)
plt.yticks([])
plt.subplot(133) # 두가지로 segmentation
# binary image를 masking으로 사용
plt.imshow(binary_img, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
plt.show()

# 확률 모델
# 몬테칼로를 이용해서 pi값을 구하시오
# 반지름이 1인 원

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

plt.title(f"Monte Carlo π Estimation : {pi_estimate:.6f}")
plt.xlim(-1,1)
plt.ylim(-1,1)

# 원 비율 유지
plt.gca().set_aspect('equal')

plt.legend()
plt.grid(True)
plt.show()