import numpy as np
arr = np.array([1, 2, 3])  # 행도 열도 아님
arr.shape

arr = np.array([1, 2, 3])  # 행도 열도 아님
arr.shape

row = np.array([[1, 2, 3]])  # 행
row.shape

col = np.array([[1],
               [2],
               [3]])
print(col.shape)

# cell magic
# % line magic
# 셀이 실행되는 시간 측정
# range : for in 문만 있어서 for ( i = 0, i < 0, i++)
# list로 리턴
a = np.arange(1000000)  # ndarray로 리턴
result = 0
for v in a:
  result += v
print("반복문을 이용한 결과 :" , result)

a = np.arange(1000000)
result = np.sum(a)
print("벡터화 연산을 이용한 결과 :", result)


# 행렬연산
import numpy as np
print(np.__version__)

print(np.eye(5))
np.identity(5)  # 단위행렬

a = np.zeros((2,3))
print(a)
a = np.ones((2, 2))
print(a)
a = np.empty((2, 2))
print(a)
a = np.full((2, 3), 7)
print(a)

a = np.linspace(0, 1, 5)
print(a)
a = np.logspace(0, 3, 4)
print(a)
a = np.random.rand(2, 2)  # 0~1 사이값
print(a)
a = np.random.randn(2, 2) # 정규분포
print(a)
a = np.diag([1, 2, 3]) # 대각선이 결정 : 크기값을 결정
print(a)

# 열우선 행렬
# 다차원배열 : 차수 2x2, datatype, 열우선 : fortran 과학계산용 언어
first = np.ndarray(shape = (2, 2), dtype = float, order = 'F')
first.fill(10)
first

print(type(np.arange(10)))

# 인덱스 0에서 시작 -1까지
array = np.arange(10).reshape(5, 2)
print("원배열 : \n", array)
b = np.ones_like(array, float)
print("\n동일 사이즈행렬 : \n", b)

# 3차원( 면행렬) - 사각형을 유지해야 함
e = np.arange(24).reshape(2, 3, 4)
print(e)
print(e.flatten()) # 1차원으로 (24개의 요소를 가짐 : 행도 아니고 열도 아닌)
e = np.arange(24).reshape(1, 1, 1, 2, 3, 4) # 차원만 증가
print("차원확대")
print(e)
print("차원축소")
e= e.reshape(-1, 4) # 열의 개수 -1은 나머지 다
print(e)
print(e[1, 2])

a = np.arange(10)
print(type(a)) # int64 : 64비트 : 4 바이트(32가 기본)
print(a.dtype) # (10,)
print(a.shape) # 1차원
print(a.ndim) #
print("아이템사이즈", a.dtype.itemsize) # 8바이트
print("타입의 이름", a.dtype.type) # int 64
print(a.size) # 10개 데이터 갯수
print(a.astype(np.float32)) # 형변환
print(a.dtype.itemsize*a.size) # 전체 메모리 사이즈


# list 하고 numpy 변환
nda = np.array([[1, 2], [3, 4]]) # list -> numoy (연산 작업)
print(type(nda))
print(nda.tolist()) # numpy -> list ( 삽입, 삭제 작업)

li = [[0, 1, 2], [3, 4, 5]]
a_2d = np.array(li)
print(a_2d)

# 문제 : 다음 데이터를 ndarray로 생성하시오
# [[11 12 13 14 15]
# [16 17 18 19 20]
# [21 22 23 24 25]
# [26 27 28 29 30]
#[31 32 33 34 35]]
# %>% , chaining(.)
a = np.arange(11, 36).reshape(5, 5)
print(a.shape)
print(a)


[[0, 1, 2, 3, 4, 5],
 [10, 11, 12, 13, 14, 15],
 [20, 21, 22, 23, 24, 25],
 [30, 31, 32, 33, 34, 35],
 [40, 41, 42, 43, 44, 45],
 [50, 51, 52, 53, 54 ,55]]

arr = []
for i in range(6):
  ad = []
  for j in range(6):
    ad.append(i * 10 + j)
  arr.append(ad)
print(arr)

np.array(arr)

arr = np.array(arr)

arr.transpose() # 전치행렬

# indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 3x3
print(arr2d.dtype) # int64
print(arr2d)
print(arr2d[:2]) # 행,  처음부터 0, 1행
print(arr2d[0]) # 0 행
print(arr2d[0][2])
print(arr2d[0, 2])

print(arr2d[:2, 1:]) # 뒤에가 없으면 끝까지
print(arr2d[1, :2])

arr2d[:2, :1] = 0 #0,1행이면서 0열에 대해서 0으로 대입
print(arr2d)
print(arr2d.ndim) # 2
# 데이터 1개가 8바이트 - 연속적 메모리에 저장
print(arr2d.strides) # (24, 8) - 다음열까지는 8바이트, 다음행 24바이트

# 연산자 오버로딩
li = [1.0, 2.0, 3.0]
li2 = li + li # * 반복
print(li2)
a = np.array([1.0, 2.0, 3.0]) # 각 요소
b = 2.0 # scala량
a * b # 각 요소에 곱함 # broadcasting

a + a # 요소 더하기

a = np.array([[1, 2],  # 2x2
             [3, 4]])
b = np.array([[4, 3, 4], # 2x3
             [2, 1, 5]])
print(a + a)
print(a * a)
# print(a + b)  # 사이즈 불일치 에러
print(a.dot(b))

# 차수 불일치 -> broadcasting
a = np.array([[0.0, 0.0, 0.0],  # 4x3 행렬
              [10.0, 10.0, 10.0],
              [20.0, 20.0, 20.0],
              [30.0, 30.0, 30.0]])
b = np.array([1.0, 2.0, 3.0])  # (3,)
a + b

b = np.array([ [1.0], [2.0], [3.0], [4.0]])
a + b

arr3d = np.arange(1, 13).reshape(2, 2, 3)
print(arr3d)
print("첫면은", arr3d[0])
print("첫면 첫행은 = ", arr3d[0][0])
print("첫면 첫행, 첫요소는 = ", arr3d[0][0][0])
# 위 데이터의 차원, 차수, 데이터타입, strides를 알아보시오
print(arr3d.ndim)
print(arr3d.shape)
print(arr3d.dtype)   #int64 -> 8바이트
print(arr3d.strides) # 데이터타입은 비트수, strides는 바이트
# 48, 24, 8바이트

# 파이썬 - call by reference 변수에는 주소
# copy는 별도의 공간에 데이터 저장
# R에서는 지연 복사 - 수정이 일어나면 복사 (파이썬도 동일)
old_values = arr3d.copy() # 포인터 위치가 달라짐
arr3d[0] = 42
print(" 값의 변경 후")
print(arr3d)
print("이전 값으로 복구")
arr3d = old_values
print(arr3d)

arr = np.arange(32).reshape((8, 4))
print(arr)
print(arr[0, 3])  # 3
print(arr[0:2, 0:2])
print("조합인덱스")
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])
# 1x5 4
# 5x3 23
print(arr[[1, 5, 7, 2]][[0, 3, 1]])
# [ 4  5  6  7]
# [ 8  9 10 11]
# [20 21 22 23 ]
print("팬시") # 행과 열로 데이터를 추출할 때
print(arr[[1, 2, 5, 7]][:, [0, 2, 3]])

# 방법 1
arr = []
for i in range(6):
  ad = []
  for j in range(6):
    ad.append(i * 10 + j)
  arr.append(ad)
arr = np.array(arr)
print(arr)

print("주황색", arr[[0,1,2,3,4], [1,2,3,4,5]])

print("빨강색", arr[[0,2,5] , 2])
# print("빨강색", arr[[0,2,5], [2,2,2]])
mask = np.array([1,0,1,0,0,1], dtype = bool)
print(arr[mask, 2])

print("하늘색(팬시로 찾기) = ", arr[3:6]   [:,[0, 2, 5]])

# 방법 2
a = np.arange(0,6)
arr = np.array([a, a+10, a+20, a+30, a+40,  a+50])
arr

names = np.array(['Seoul', 'Daejun', 'chungju', 'Seoul', 'chungju', 'Daejun', 'Daejun' ])
data = np.random.randn(7, 4)
print(data)
print(names =='Seoul')
print(data[names == 'Seoul'])  # 2개의 행 선택
print(data[names == 'Seoul', 2:]) # 2x2
print(data[names == 'Seoul', 3]) # 2x1
print(names != 'Seoul')  # 5개행, 개별부정
print(~(names == 'Seoul')) # 전체부정
mask = (names == 'Seoul') | (names == 'chungju')  # &:and
print(mask)
print(data[mask])

# 문제 : 첫번째 열이 0.5보다 큰 행을 선택하시오
print(data[:, 0]> 0.5)
print(data[data[:,0]>0.5])

# 3항 연산자
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

print(zip(xarr, yarr, cond))
print(list(zip(xarr, yarr, cond)))
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
print("result = ", result)

# for 문이 자동 : 벡터화 연산
result = np.where(cond, xarr, yarr)
print(result)

# 문제 :  다음 배열요소에 대해서 0보다 크면 2 아니면 -2로 표현하시오
arr = np.random.randn(4, 4)
print(arr)


#result = [(i>=0 if 2 else  -2) for i in arr]
#print("result =", result)

print(np.where(arr > 0, 2, -2))

# 그물망 - grid 좌표계  조합을 다 구성
x, y = np.meshgrid(  # 2차원, 3차원 시각화할 때 사용
    np.arange(3),
    np.arange(4)
)
x, y

x, y = np.ogrid[:3, :4] # 행과 열 인덱스
print(x)
print(y)

# 5x5 meshgrid보다 메모리에서 유리
x, y = np.ogrid[-2:3, -2:3]
mask = x **2 + y**2 <= 4 # 원의 공식
print(mask)

# 3항 연산자를 이용해서 처리하시오
cond1 = np.array([1, 0, 1, 1, 0, 0, 1], dtype = bool)
cond2 = np.array([0, 1, 1, 1, 0, 1, 1], dtype = bool)
# 위 두개의 데이터에 대해서 두개가 다 참이면 0, cond1이 참이면 1, #cond2가 참이면 2 그 외는 3으로 처리하여 출력하시오


result = []
for i in range(len(cond1)):
  if(cond1[i] and cond2[i]):
    result.append(0)
  elif cond1[i]:
    result.append(1)
  elif cond2[i]:
    result.append(2)
  else:
    result.append(3)
print("조건문", result)

# n-1 회 비교
result = np.where(cond1 & cond2 , 0, np.where(cond1, 1, np.where(cond2, 2, 3)))
print("3항 연산자", result)

result = 1*(cond1 & ~cond2) + 2*(cond2 & ~ cond1) + 3 * ~(cond1 | cond2)
print("수식을 이용한",  result)

# np.newaxis 차원일치를 위해서
arr = np.arange(1, 10)
print(arr.shape)
print(arr)
# : 모든 데이터
row_vec = arr[np.newaxis, np.newaxis, np.newaxis, :]
print("차수", row_vec.shape)
print(row_vec)
# 행을 열로 만들 때
col_vec = arr[:, np.newaxis, np.newaxis, np.newaxis]
print(col_vec.shape)
print(col_vec)

a = np.array([0, 1, 2, 3, 4])
print(a[::2])

a1 = np.array([[5,6],
              [7,8]])
ra = a1.ravel()  # 1차원
re = a1.reshape(-1) # 1차원
fl = a1.flatten()  # 수정이 안됨 : 복사해서 처리했다.
# python은 view를 이용 (포인터를 이용해서 같은 장소)
# 지연복사 flatten은 처음 복사해서 작업
print(ra, re, fl)
a1[0][0] = 100  # 값을 변경
print(ra, re, fl)

# from numpy import math
# np 고속,
import math
print(np.ceil(1.001)) # 올림
print(np.floor(1.001)) # 내림
print(math.factorial(5))
print(np.gcd(10, 125)) # 최대공약수
print(np.trunc(1.001)) # 음수에서 소수점 이하는 버림
print(np.trunc(1.999))
print(max(1, 2, 3, 4)) # python core

# 삼각함수 : 회전, 주기성 있는 데이터
# 미분이 어려워 -> 오일러정리 -> exp 로 변환한 다음 계산
# # 0, 90, 180 컴퓨터에서는 각도가 아니라 라디안으로 사용
print(np.pi / 2) # 90 -> 반지름이 1일 때의 호의 길이
theta = np.linspace(0, np.pi, 3) # 0, 90, 180
print("theta      =", theta)
# 좌표 (x축 cos, y축 sin)
print("sin(theta) =", np.sin(theta)) # 0
print("cos(theta) =", np.cos(theta)) # 1
# vector 내적 : 내적 = anorm * bnorm * cos(theta)
# cos 유사도
# 내적이 0이면 직교다
print("tan(theta) =", np.tan(theta))

print(np.radians([0, 90, 180]))
# 삼각함수 : 각도를 길이의 비로 바꿔줌

# 길이의 비 => 각도 (라디안으로 표현)
x = [ -1, 0, 1]
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

# 호의 길이를 각도로
np.degrees(np.pi)

# 문제 : 밑변이 5이고 그 사이각이 60도라면 대각선의 길이는 얼마인가?
# cos.theta = 밑변 / 빗변
res = np.cos(np.radians(60)) # = 밑변 / 빗변
# 0.5 = 5 / rad
rad = 5 / res
rad # 대각선의 길의는 10

# 대각선이 8이고 그 사이각이 45도라면 밑변의 길이와 높이를 구하시오
# sin theta = 높이 / 빗변
sintheta = np.sin(np.radians(45))
8 * sintheta

costheta = np.cos(np.radians(45))
8 * costheta

# exp : 기하급수적 증가, 감소, 차이값을 극대화 (분류 : softmax)
# 정규분포에서 x축 : 표준편차
# y축 밀도 : 이산그래프에서 개수와 같은 의미
x = [1, 2, 3]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("e^x   =", np.exp(x))  # 자연지수 2.718 자기증가 (자연현상)
print("2^x   =", np.exp2(x)) # 정보관련 처리 => 256가지 표현
print("10^x  =", np.power(10, x)) # 금융,

#matplotlib는 numpy를 시각화
import matplotlib.pyplot as plt
y_exp = np.exp(x) # 자연지수
y_exp2 = np.exp2(x) # 밑2 exponent
y_power10 = np.power(10, x)
plt.figure(figsize = (8, 5)) # 도화지
plt.plot(x, y_exp, marker = 'o', label = 'e^x = np.exp(x)')
plt.plot(x, y_exp2, marker = 's', label = '2^x = np.exp2(x)')
plt.plot(x, y_power10, marker = '^', label = '10^x = np.power(10, x)')
plt.xlabel("x") # label
plt.ylabel("Value")
plt.title("Exponential Functions") # 제목
plt.legend()  # 범례
plt.grid(True)  # 격자 - 위치 정확하게
output_path = "./exponential_comparison.png"
plt.savefig(output_path, bbox_inches = 'tight')
print("그래프 저장 완료:", output_path)

# 자연대수에 몇승을 해야 1이 되나
print("ln(x)     =", np.log(x)) # 2.718 밑수에 1.09861229 배를 하면 3이 된다
print("log2(x)   =", np.log2(x))
print("log10(x)  =", np.log10(x))

# seaborn 분석용 시각화 패키지 , matplotlib를 배경으로 함
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
X = np.random.rand(10, 2) # 0 ~ 1 사이값
print(X)
plt.scatter(X[:, 0], X[:, 1], s = 100);

# 방정식의 해 구하기
# 3 * x0 + x1 = 9
# x0 + 2 * x1 = 8
# [3, 1]  [x0]    = [9, 8]
# [1, 2]  [x1]
a = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(a, b)
print("방정식의 해 = ", x)
3 * 2 + 1 * 3
1 * 2 +  2 * 3

# 2x + y + z = 1
# 4x + 3y + 4z = 2
# -4x + 2y + 2z = -6
a = [[2, 1, 1], [4, 3, 4], [-4, 2, 2]] # 행렬은 나눗셈이 없어서 역행렬
b = [1, 2, -6]
x = np.linalg.solve(a, b) # 역행렬을 구하고 행렬곱으로 문제를 해결
print("방정식의 해", x)
2 * 1 + (1 * -2) + 1 # 검산

# 정방행렬만 역행력을 구할 수 있음, 비정방행렬인 경우는 의사(유사)역행렬을 구함(pinv)
import numpy as np
a = np.random.randn(9, 6)
B = np.linalg.pinv(a) # 유사 역행렬은 ABA를 만족
print(np.allclose(a, np.dot(a, np.dot(B, a))))

# 2차 방정식 : 최저점이 최적점
# 미분이 0이면 최적점
def my_func(x) :
  return 3*x**2 + 2*x + 1
x = np.linspace(-3, 3)
y2 = my_func(x) # 변곡점 1개
y3 = x**3 - 3*x**2 + 2*x + 1 # 변곡점 2개
y4 = x**4 - 4*x**2 + 3
y = y4 # 변곡점 3개 -> 지역해 빠진다 -> 경사하강법
plt.plot(x, y)
plt.plot(x, y, 'bo')
plt.xlabel("x", size = 14)
plt.ylabel("y", size = 14)
plt.grid
plt.show()

row = np.array([[1, 2, 3]])  # 행
row.shape

col = np.array([[1],
               [2],
               [3]])
print(col.shape)

# cell magic
# % line magic
# 셀이 실행되는 시간 측정
# range : for in 문만 있어서 for ( i = 0, i < 0, i++)
# list로 리턴
a = np.arange(1000000)  # ndarray로 리턴
result = 0
for v in a:
  result += v
print("반복문을 이용한 결과 :" , result)

a = np.arange(1000000)
result = np.sum(a)
print("벡터화 연산을 이용한 결과 :", result)


# 행렬연산
import numpy as np
print(np.__version__)

print(np.eye(5))
np.identity(5)  # 단위행렬

a = np.zeros((2,3))
print(a)
a = np.ones((2, 2))
print(a)
a = np.empty((2, 2))
print(a)
a = np.full((2, 3), 7)
print(a)

a = np.linspace(0, 1, 5)
print(a)
a = np.logspace(0, 3, 4)
print(a)
a = np.random.rand(2, 2)  # 0~1 사이값
print(a)
a = np.random.randn(2, 2) # 정규분포
print(a)
a = np.diag([1, 2, 3]) # 대각선이 결정 : 크기값을 결정
print(a)

# 열우선 행렬
# 다차원배열 : 차수 2x2, datatype, 열우선 : fortran 과학계산용 언어
first = np.ndarray(shape = (2, 2), dtype = float, order = 'F')
first.fill(10)
first

print(type(np.arange(10)))

# 인덱스 0에서 시작 -1까지
array = np.arange(10).reshape(5, 2)
print("원배열 : \n", array)
b = np.ones_like(array, float)
print("\n동일 사이즈행렬 : \n", b)

# 3차원( 면행렬) - 사각형을 유지해야 함
e = np.arange(24).reshape(2, 3, 4)
print(e)
print(e.flatten()) # 1차원으로 (24개의 요소를 가짐 : 행도 아니고 열도 아닌)
e = np.arange(24).reshape(1, 1, 1, 2, 3, 4) # 차원만 증가
print("차원확대")
print(e)
print("차원축소")
e= e.reshape(-1, 4) # 열의 개수 -1은 나머지 다
print(e)
print(e[1, 2])

a = np.arange(10)
print(type(a)) # int64 : 64비트 : 4 바이트(32가 기본)
print(a.dtype) # (10,)
print(a.shape) # 1차원
print(a.ndim) #
print("아이템사이즈", a.dtype.itemsize) # 8바이트
print("타입의 이름", a.dtype.type) # int 64
print(a.size) # 10개 데이터 갯수
print(a.astype(np.float32)) # 형변환
print(a.dtype.itemsize*a.size) # 전체 메모리 사이즈


# list 하고 numpy 변환
nda = np.array([[1, 2], [3, 4]]) # list -> numoy (연산 작업)
print(type(nda))
print(nda.tolist()) # numpy -> list ( 삽입, 삭제 작업)

li = [[0, 1, 2], [3, 4, 5]]
a_2d = np.array(li)
print(a_2d)

# 문제 : 다음 데이터를 ndarray로 생성하시오
# [[11 12 13 14 15]
# [16 17 18 19 20]
# [21 22 23 24 25]
# [26 27 28 29 30]
#[31 32 33 34 35]]
# %>% , chaining(.)
a = np.arange(11, 36).reshape(5, 5)
print(a.shape)
print(a)


[[0, 1, 2, 3, 4, 5],
 [10, 11, 12, 13, 14, 15],
 [20, 21, 22, 23, 24, 25],
 [30, 31, 32, 33, 34, 35],
 [40, 41, 42, 43, 44, 45],
 [50, 51, 52, 53, 54 ,55]]

arr = []
for i in range(6):
  ad = []
  for j in range(6):
    ad.append(i * 10 + j)
  arr.append(ad)
print(arr)

np.array(arr)

arr = np.array(arr)

arr.transpose() # 전치행렬

# indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 3x3
print(arr2d.dtype) # int64
print(arr2d)
print(arr2d[:2]) # 행,  처음부터 0, 1행
print(arr2d[0]) # 0 행
print(arr2d[0][2])
print(arr2d[0, 2])

print(arr2d[:2, 1:]) # 뒤에가 없으면 끝까지
print(arr2d[1, :2])

arr2d[:2, :1] = 0 #0,1행이면서 0열에 대해서 0으로 대입
print(arr2d)
print(arr2d.ndim) # 2
# 데이터 1개가 8바이트 - 연속적 메모리에 저장
print(arr2d.strides) # (24, 8) - 다음열까지는 8바이트, 다음행 24바이트

# 연산자 오버로딩
li = [1.0, 2.0, 3.0]
li2 = li + li # * 반복
print(li2)
a = np.array([1.0, 2.0, 3.0]) # 각 요소
b = 2.0 # scala량
a * b # 각 요소에 곱함 # broadcasting

a + a # 요소 더하기

a = np.array([[1, 2],  # 2x2
             [3, 4]])
b = np.array([[4, 3, 4], # 2x3
             [2, 1, 5]])
print(a + a)
print(a * a)
# print(a + b)  # 사이즈 불일치 에러
print(a.dot(b))

# 차수 불일치 -> broadcasting
a = np.array([[0.0, 0.0, 0.0],  # 4x3 행렬
              [10.0, 10.0, 10.0],
              [20.0, 20.0, 20.0],
              [30.0, 30.0, 30.0]])
b = np.array([1.0, 2.0, 3.0])  # (3,)
a + b

b = np.array([ [1.0], [2.0], [3.0], [4.0]])
a + b

arr3d = np.arange(1, 13).reshape(2, 2, 3)
print(arr3d)
print("첫면은", arr3d[0])
print("첫면 첫행은 = ", arr3d[0][0])
print("첫면 첫행, 첫요소는 = ", arr3d[0][0][0])
# 위 데이터의 차원, 차수, 데이터타입, strides를 알아보시오
print(arr3d.ndim)
print(arr3d.shape)
print(arr3d.dtype)   #int64 -> 8바이트
print(arr3d.strides) # 데이터타입은 비트수, strides는 바이트
# 48, 24, 8바이트

# 파이썬 - call by reference 변수에는 주소
# copy는 별도의 공간에 데이터 저장
# R에서는 지연 복사 - 수정이 일어나면 복사 (파이썬도 동일)
old_values = arr3d.copy() # 포인터 위치가 달라짐
arr3d[0] = 42
print(" 값의 변경 후")
print(arr3d)
print("이전 값으로 복구")
arr3d = old_values
print(arr3d)

arr = np.arange(32).reshape((8, 4))
print(arr)
print(arr[0, 3])  # 3
print(arr[0:2, 0:2])
print("조합인덱스")
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])
# 1x5 4
# 5x3 23
print(arr[[1, 5, 7, 2]][[0, 3, 1]])
# [ 4  5  6  7]
# [ 8  9 10 11]
# [20 21 22 23 ]
print("팬시") # 행과 열로 데이터를 추출할 때
print(arr[[1, 2, 5, 7]][:, [0, 2, 3]])

# 방법 1
arr = []
for i in range(6):
  ad = []
  for j in range(6):
    ad.append(i * 10 + j)
  arr.append(ad)
arr = np.array(arr)
print(arr)

print("주황색", arr[[0,1,2,3,4], [1,2,3,4,5]])

print("빨강색", arr[[0,2,5] , 2])
# print("빨강색", arr[[0,2,5], [2,2,2]])
mask = np.array([1,0,1,0,0,1], dtype = bool)
print(arr[mask, 2])

print("하늘색(팬시로 찾기) = ", arr[3:6]   [:,[0, 2, 5]])

# 방법 2
a = np.arange(0,6)
arr = np.array([a, a+10, a+20, a+30, a+40,  a+50])
arr

names = np.array(['Seoul', 'Daejun', 'chungju', 'Seoul', 'chungju', 'Daejun', 'Daejun' ])
data = np.random.randn(7, 4)
print(data)
print(names =='Seoul')
print(data[names == 'Seoul'])  # 2개의 행 선택
print(data[names == 'Seoul', 2:]) # 2x2
print(data[names == 'Seoul', 3]) # 2x1
print(names != 'Seoul')  # 5개행, 개별부정
print(~(names == 'Seoul')) # 전체부정
mask = (names == 'Seoul') | (names == 'chungju')  # &:and
print(mask)
print(data[mask])

# 문제 : 첫번째 열이 0.5보다 큰 행을 선택하시오
print(data[:, 0]> 0.5)
print(data[data[:,0]>0.5])

# 3항 연산자
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

print(zip(xarr, yarr, cond))
print(list(zip(xarr, yarr, cond)))
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
print("result = ", result)

# for 문이 자동 : 벡터화 연산
result = np.where(cond, xarr, yarr)
print(result)

# 문제 :  다음 배열요소에 대해서 0보다 크면 2 아니면 -2로 표현하시오
arr = np.random.randn(4, 4)
print(arr)


#result = [(i>=0 if 2 else  -2) for i in arr]
#print("result =", result)

print(np.where(arr > 0, 2, -2))

# 그물망 - grid 좌표계  조합을 다 구성
x, y = np.meshgrid(  # 2차원, 3차원 시각화할 때 사용
    np.arange(3),
    np.arange(4)
)
x, y

x, y = np.ogrid[:3, :4] # 행과 열 인덱스
print(x)
print(y)

# 5x5 meshgrid보다 메모리에서 유리
x, y = np.ogrid[-2:3, -2:3]
mask = x **2 + y**2 <= 4 # 원의 공식
print(mask)

# 3항 연산자를 이용해서 처리하시오
cond1 = np.array([1, 0, 1, 1, 0, 0, 1], dtype = bool)
cond2 = np.array([0, 1, 1, 1, 0, 1, 1], dtype = bool)
# 위 두개의 데이터에 대해서 두개가 다 참이면 0, cond1이 참이면 1, #cond2가 참이면 2 그 외는 3으로 처리하여 출력하시오


result = []
for i in range(len(cond1)):
  if(cond1[i] and cond2[i]):
    result.append(0)
  elif cond1[i]:
    result.append(1)
  elif cond2[i]:
    result.append(2)
  else:
    result.append(3)
print("조건문", result)

# n-1 회 비교
result = np.where(cond1 & cond2 , 0, np.where(cond1, 1, np.where(cond2, 2, 3)))
print("3항 연산자", result)

result = 1*(cond1 & ~cond2) + 2*(cond2 & ~ cond1) + 3 * ~(cond1 | cond2)
print("수식을 이용한",  result)

# np.newaxis 차원일치를 위해서
arr = np.arange(1, 10)
print(arr.shape)
print(arr)
# : 모든 데이터
row_vec = arr[np.newaxis, np.newaxis, np.newaxis, :]
print("차수", row_vec.shape)
print(row_vec)
# 행을 열로 만들 때
col_vec = arr[:, np.newaxis, np.newaxis, np.newaxis]
print(col_vec.shape)
print(col_vec)

a = np.array([0, 1, 2, 3, 4])
print(a[::2])

a1 = np.array([[5,6],
              [7,8]])
ra = a1.ravel()  # 1차원
re = a1.reshape(-1) # 1차원
fl = a1.flatten()  # 수정이 안됨 : 복사해서 처리했다.
# python은 view를 이용 (포인터를 이용해서 같은 장소)
# 지연복사 flatten은 처음 복사해서 작업
print(ra, re, fl)
a1[0][0] = 100  # 값을 변경
print(ra, re, fl)

# from numpy import math
# np 고속,
import math
print(np.ceil(1.001)) # 올림
print(np.floor(1.001)) # 내림
print(math.factorial(5))
print(np.gcd(10, 125)) # 최대공약수
print(np.trunc(1.001)) # 음수에서 소수점 이하는 버림
print(np.trunc(1.999))
print(max(1, 2, 3, 4)) # python core

# 삼각함수 : 회전, 주기성 있는 데이터
# 미분이 어려워 -> 오일러정리 -> exp 로 변환한 다음 계산
# # 0, 90, 180 컴퓨터에서는 각도가 아니라 라디안으로 사용
print(np.pi / 2) # 90 -> 반지름이 1일 때의 호의 길이
theta = np.linspace(0, np.pi, 3) # 0, 90, 180
print("theta      =", theta)
# 좌표 (x축 cos, y축 sin)
print("sin(theta) =", np.sin(theta)) # 0
print("cos(theta) =", np.cos(theta)) # 1
# vector 내적 : 내적 = anorm * bnorm * cos(theta)
# cos 유사도
# 내적이 0이면 직교다
print("tan(theta) =", np.tan(theta))

print(np.radians([0, 90, 180]))
# 삼각함수 : 각도를 길이의 비로 바꿔줌

# 길이의 비 => 각도 (라디안으로 표현)
x = [ -1, 0, 1]
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

# 호의 길이를 각도로
np.degrees(np.pi)

# 문제 : 밑변이 5이고 그 사이각이 60도라면 대각선의 길이는 얼마인가?
# cos.theta = 밑변 / 빗변
res = np.cos(np.radians(60)) # = 밑변 / 빗변
# 0.5 = 5 / rad
rad = 5 / res
rad # 대각선의 길의는 10

# 대각선이 8이고 그 사이각이 45도라면 밑변의 길이와 높이를 구하시오
# sin theta = 높이 / 빗변
sintheta = np.sin(np.radians(45))
8 * sintheta

costheta = np.cos(np.radians(45))
8 * costheta

# exp : 기하급수적 증가, 감소, 차이값을 극대화 (분류 : softmax)
# 정규분포에서 x축 : 표준편차
# y축 밀도 : 이산그래프에서 개수와 같은 의미
x = [1, 2, 3]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("e^x   =", np.exp(x))  # 자연지수 2.718 자기증가 (자연현상)
print("2^x   =", np.exp2(x)) # 정보관련 처리 => 256가지 표현
print("10^x  =", np.power(10, x)) # 금융,

#matplotlib는 numpy를 시각화
import matplotlib.pyplot as plt
y_exp = np.exp(x) # 자연지수
y_exp2 = np.exp2(x) # 밑2 exponent
y_power10 = np.power(10, x)
plt.figure(figsize = (8, 5)) # 도화지
plt.plot(x, y_exp, marker = 'o', label = 'e^x = np.exp(x)')
plt.plot(x, y_exp2, marker = 's', label = '2^x = np.exp2(x)')
plt.plot(x, y_power10, marker = '^', label = '10^x = np.power(10, x)')
plt.xlabel("x") # label
plt.ylabel("Value")
plt.title("Exponential Functions") # 제목
plt.legend()  # 범례
plt.grid(True)  # 격자 - 위치 정확하게
output_path = "./exponential_comparison.png"
plt.savefig(output_path, bbox_inches = 'tight')
print("그래프 저장 완료:", output_path)

# 자연대수에 몇승을 해야 1이 되나
print("ln(x)     =", np.log(x)) # 2.718 밑수에 1.09861229 배를 하면 3이 된다
print("log2(x)   =", np.log2(x))
print("log10(x)  =", np.log10(x))

# seaborn 분석용 시각화 패키지 , matplotlib를 배경으로 함
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
X = np.random.rand(10, 2) # 0 ~ 1 사이값
print(X)
plt.scatter(X[:, 0], X[:, 1], s = 100);

# 방정식의 해 구하기
# 3 * x0 + x1 = 9
# x0 + 2 * x1 = 8
# [3, 1]  [x0]    = [9, 8]
# [1, 2]  [x1]
a = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(a, b)
print("방정식의 해 = ", x)
3 * 2 + 1 * 3
1 * 2 +  2 * 3

# 2x + y + z = 1
# 4x + 3y + 4z = 2
# -4x + 2y + 2z = -6
a = [[2, 1, 1], [4, 3, 4], [-4, 2, 2]] # 행렬은 나눗셈이 없어서 역행렬
b = [1, 2, -6]
x = np.linalg.solve(a, b) # 역행렬을 구하고 행렬곱으로 문제를 해결
print("방정식의 해", x)
2 * 1 + (1 * -2) + 1 # 검산

# 정방행렬만 역행력을 구할 수 있음, 비정방행렬인 경우는 의사(유사)역행렬을 구함(pinv)
import numpy as np
a = np.random.randn(9, 6)
B = np.linalg.pinv(a) # 유사 역행렬은 ABA를 만족
print(np.allclose(a, np.dot(a, np.dot(B, a))))

# 2차 방정식 : 최저점이 최적점
# 미분이 0이면 최적점
def my_func(x) :
  return 3*x**2 + 2*x + 1
x = np.linspace(-3, 3)
y2 = my_func(x) # 변곡점 1개
y3 = x**3 - 3*x**2 + 2*x + 1 # 변곡점 2개
y4 = x**4 - 4*x**2 + 3
y = y4 # 변곡점 3개 -> 지역해 빠진다 -> 경사하강법
plt.plot(x, y)
plt.plot(x, y, 'bo')
plt.xlabel("x", size = 14)
plt.ylabel("y", size = 14)
plt.grid
plt.show()