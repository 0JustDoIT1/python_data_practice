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

# PCA vs SVD (singula value decomposition) 비교
import numpy as np
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
eigvals, eigvecs = np.linalg.eig(X.T @ X)
idx = np.argsort(eigvals)[::-1] # 순서가 변경되어서
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
# 행렬곱은 전치 (거듭제곱할 때처럼)
U, S, Vt = np.linalg.svd(X)
V = Vt.T # 전치
# + -> -로 표현
for i in range(V.shape[1]):
  if np.dot(eigvecs[:, i], V[:, i]) < 0:
    V[:, i] *= -1

print("eigvecs")
print(eigvecs)
print("V")
print(V)
print("같은가?")
print(np.allclose(eigvecs, V))

# SVD (singular value decomposition) 특이행렬분해
# 행렬분해의 목적 : 계산을 간단하게 하기 위함
# PCA : 정방행렬, SVD : 비정방행렬
# 희소행렬의 특징 추출, 잠재변수 (text mining), 주성분분석
from scipy import linalg
A = np.array([[1,2,3], [4,5,6]])
A

M, N = A.shape
M, N # 2 x 3 비정방행렬

U, s, Vh = linalg.svd(A) # m x n : 2 x 3
U, s, Vh # 왼쪽 특이행렬, 고유치, 오른쪽 특이행렬
# 고유치는 PCA의 절반값

Sig = linalg.diagsvd(s, M, N)
Sig

U @ (Sig @ Vh) # Sig가 두번 곱해지기 때문에 절반 사이즈로

# TruncatedSVD 는 원본 데이터를 사용 : 평균을 빼지 않음
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
np.random.seed(0)
X_dense = np.random.rand(100, 100) # 100 x 100
X_dense[:, 2 * np.arange(50)] = 0
print(X_dense)
# 0이 많은 행렬에 -1
# 텍스트 분석 : 0 이상의 개수에 -1 -> 음수
# 장바구니분석
X = csr_matrix(X_dense) # Compressed 압축형태의 희소행렬
# 100 -> 5개로 차원 축소
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)
print("분산설명", svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)

# term-document matrix
# 잠재변수 (topic)
#             '문서 1'    '문서 2'    '문서 3'
#   AI        3           0           2
#   ML        2           3           0
#   데이터    0           3           5
# SVD를 적용하면 잠재-topic을 발견
A = np.array([
    [3, 0, 2],
    [2, 3, 0],
    [0, 3, 5]
])
words = ["AI", "ML", "데이터"]
documents = ["문서 1", "문서 2", "문서 3"]
print("--- 1. 원본 단어-문서 행렬 (A) ---")
print(A)

U, s, VT = np.linalg.svd(A) # 3 x 3
Sigma = np.diag(s)
print("특이값 (s):", s.round(4))
print("행렬 U (단어-토픽):\n", U.round(4))
# 행이 단어
#           topic1(AI, ML)      topic 2(ML특징에 AI 첨가)       topic 3(AI긍정)
# AI        -0.3701             0.5475                          0.7505
# ML        -0.3454             0.6688                          -0.6583
# 데이터    -0.8624             -0.5029                         -0.0584
print("행렬 VT (토픽-문서):\n", VT.round(4))

# 단어를 이용해서는 topic의 의미를 찾고
# 문서는 어떤 토픽이 강한 영향을 미쳤나
# 오른쪽 특이행렬
#           문서1(ML에 AI를 첨가문서)       문서2(AI를 부정 내용)        문서3(AI, ML에 관한 문서)
# topic1    -0.2783                         -0.5598                      -0.7805
# topic2    -0.3454                         0.6688                       -0.6583
# topic3    -0.8624                         -0.5029                      -0.0584

# PCA를 이용한 시각화
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
iris = sns.load_dataset("iris")
print(iris.head())
print(iris.shape)
# 원데이터 변수 -> 4개
# 상관도 -> 2차원
sns.pairplot(iris, hue='species', height=1.5)
X_iris = iris.iloc[:,:4]
from sklearn.decomposition import PCA
model = PCA(n_components=2) # 4 -> 2 (변수를 2개 선택하는 것과의 차이)
# 상관계수행렬 -> 고유값 분해 -> 주성분 선택 (2)
# 주성분축은 모든 변수가 참여해서 만들기 때문에
# 2개의 변수가 4개의 변수를 대변할 수 있기 때문
model.fit(X_iris)
X_2D = model.transform(X_iris)
print(X_2D.shape)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot(x="PCA1", y="PCA2", hue="species", data=iris, fit_reg=True)
plt.show()

# PCA 모델 -> 내부적 SVD
model.singular_values_ # 특이도 (다른 말로 고유치 / 2)

model.explained_variance_

model.explained_variance_ratio_ # 설명비율

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise  * np.random.randn(m)
print('X.shape:', X.shape)
X

# 공분산
# (x1 - xbar)(y1 - ybar) + (x2 - xbar)(y2 - ybar) ......
# ----------------------------------------------------------
#                       (n-1)
X.mean(axis=0) # 열평균

X - X.mean(axis=0)

X_cen = X - X.mean(axis=0) # 5점

# 내적을 이용해서
# 입력 순서가 중요
# 3x60 60x3 -> 3x3 공분산값 (평균을 뺀 값들을 만들어 놓고 계산)
X_cov=np.dot(X_cen.T, X_cen) / (60 - 1)
X_cov

w, v = np.linalg.eig(X_cov)
print("eigenvalue :", w)
print("eigenvector :\n", v)

np.sqrt(np.sum(v[1]**2)) # 고유벡터는 축 : 축은 방향
# 방향은 정규화 -> 사이즈가 1

np.sqrt(np.sum(v[:,1]**2))

print("설명력 : ", w/w.sum())

U, D, V_t = np.linalg.svd(X_cen) # 60x3
print(U.shape, D.shape, V_t.shape)

print('explained variance ratio :', D ** 2 / np.sum(D ** 2))

V_t.T

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X_cen)

# svd 의 D값과 같음
print('singular value :', pca.singular_values_)

# 고유벡터와 동일
print('singular vector :\n', pca.components_.T)

# 손글씨 데이터
import tensorflow as tf # DL 툴

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x.shape

train_y.shape

28 * 28

from sklearn.decomposition import PCA
# (60000, 28, 28)
# => 60000x784
# 모든 학습은 일렬로 서서 입장
# fully-connected 한다
train_x = train_x.reshape(-1, 28*28) # -1은 전체다
pca = PCA(n_components=0.95) # 분산의 설명한계
X_reduced = pca.fit_transform(train_x)
print(pca.n_components_) # 784 -> 154 주성분
# 784x784 -> 784x154
# 원래 데이터 60000x784 @ 784x154 -> 60000x154

X_reduced.shape # 학습 데이터
# 784x154의 역행렬을 구해서 곱해주면

X_recovered = pca.inverse_transform(X_reduced) # 154x784
# 60000 x 154 @ 154 x 784 -> 60000 x 784
X_recovered.shape

# train_x 와 X_recovered 복원된 데이터와의 차이점은
# train_x : origin, X_recovered 복원된 데이터 (노이즈가 제거된 데이터)

import matplotlib
def plot_digits(instances, images_per_row=5, **options):
  size=28
  images_per_row = min(len(instances), images_per_row)
  images = [instance.reshape(size, size) for instance in instances]
  n_rows = (len(instances) - 1) // images_per_row + 1
  row_images = []
  n_empty = n_rows * images_per_row - len(instances)
  images.append(np.zeros((size, size * n_empty)))
  for row in range(n_rows):
    rimages = images[row * images_per_row : (row + 1) * images_per_row]
    row_images.append(np.concatenate(rimages, axis=1))
  image = np.concatenate(row_images, axis=0)
  plt.imshow(image, cmap = matplotlib.cm.binary, **options)
  plt.axis("off")

import numpy as np
plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(train_x[::2100])
plt.title("original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("no noise", fontsize=16)
plt.show()
# 0을 0으로 하지 못함
# PCA는 차원을 축소하더라도 원본 유지(noise만 제거)
# 28x28 -> 784(origin) -> PCA(154) -> 784(no noise)

# incrementalPCA

from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(train_x, n_batches):
  print(".", end="")
  inc_pca.partial_fit(X_batch) # 부분학습
X_inc_reduced = inc_pca.transform(train_x) # 고유벡터(주성분으로 차원축소)

X_recovered_inc_pca = inc_pca.inverse_transform(X_inc_reduced)
X_recovered_inc_pca.shape

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(train_x[::2100])
plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()

# 저장 : 메모리와 하드디스크 직접 연결
filename = "my_mnist.data"
m, n = train_x.shape # 차수
X_mm = np.memmap(filename, dtype="float32", mode="write", shape=(m, n))
X_mm[:] = train_x # 자동 저장

del X_mm

# 로딩 (메모리 사이즈보다 큰 데이터를 처리하기 위함)
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
batch_size = m // n_batches # 100
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

import time
for n_components in (2, 10, 154):
  print("n_components =", n_components)
  regular_pca = PCA(n_components=n_components, svd_solver="full") # 기본
  inc_pca = IncrementalPCA(n_components=n_components, batch_size=500) # 증분
  rnd_pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized") # random PCA : sampling으로 간략하게 고유치와 고유벡터를 구해서 작업
  for name, pca in (("PCA", regular_pca), ("Inc PCA", inc_pca), ("Rnd PCA", rnd_pca)):
    t1 = time.time()
    pca.fit(train_x)
    t2 = time.time()
    print("{}: {:.1f} seconds".format(name, t2 - t1))

# KernelPCA
# 3차원
# moon 반달 모양

from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
X

# Kernel : 방사형 커널, polynomial, sigmoid, linear, cosine
# Kernel 적용 : 비선형관계를 파악하기 위해 고차원으로 보냄
# 실제 데이터를 고차원으로 보내는 것이 아니라
# 함수를 이용해서 계산한 다음 데이터 간의 유사도 행렬을 만듬
# PCA 공분산 행렬, KernelPCA는 유사도 행렬 (비선형관계)
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components = 2, kernel="cosine", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.Spectral, marker='o')
ax1.set_title("스위스롤(3D)")
ax1.set_xlabel('X0')
ax1.set_ylabel('X1')
ax1.set_zlabel('X2')
ax1.view_init(elev=10, azim=-60)
ax2 = fig.add_subplot(122)
scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.Spectral, marker='o')
ax2.set_title('Kernel PCA (RBF)(2D)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.grid(True)
fig.colorbar(scatter2, ax=ax2, label='Intrinsic Parameter (t)')
plt.tight_layout()
plt.show()

# 모든 문서에 있는 단어들을 취합
# 불용어를 제거
# 명사, 형용사, 동사 -> 전처리
# -------------Machine-------------
# 1번 문서 :  1*log(4/2) # 총 문서 수 / Machine이 들어 있는 문서의 개수
documents = ["Machine learning is powerful.",
             "AI and deep learning are evolving.",
             "Data science uses AI and machine learning.",
             "Natural language processing is a subfield of AI."]
# 단어취합
# 각 문서에 있는 단어들을 단어 번호로 매칭
# 카운트
# TfidfVectorizer
# text frequency 단어 빈도
# idf : inverse document frequency 단어의 중요도까지 고려
# 중요하지 않은 단어들은 모든 문서에 나타난다
# 문서에서 벡터를 만드는 과정을 embedding 한다
# 단어 -> vector [0000010101: AI]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA # L1 규제 : 변수를 0
vectorizer = TfidfVectorizer() # 단어를 취합하고 tfidf를 구하고
X_tfidf = vectorizer.fit_transform(documents).toarray()
# L1 규제를 적용 -> 변수를 0으로 가게 하는 역할
# 중요한 변수(단어)만 선택
# 문서를 2개의 주제(topic)로 차원 축소
spca = SparsePCA(n_components=2, alpha=0.2, random_state=42)
X_spca = spca.fit_transform(X_tfidf)

print("텍스트 데이터 차원 축소 후:", X_spca.shape)

# 17개의 단어로 tfidf, 없는 것은 단어 무
X_tfidf # 4개의 문서가 17개의 단어로 표현

X_spca # 2개의 topic으로 차원 축소

print("SparsePCA 성분 로딩 (단어 기여도):")
components = spca.components_
# 단어는 들어 오면서 index로 변경
# i am a boy
# a, am, boy, i
# 0  1   2    3 -> 숫자화
feature_names = vectorizer.get_feature_names_out()
for i, component in enumerate(components):
  # component를 구성하는 요소 3개만 추출
  top_indices = np.argsort(np.abs(component))[::-1][:3]
  top_features = [(feature_names[j], component[j].round(3))
  for j in top_indices]

  print(f"Component {i+1} (Topic {i+1}):")
  print(f"Top Feautres: {top_features}")

components # topic에 기여한 변수들

feature_names # 단어장 dictionary 사전

# MDS (multi dimmension scale) : 다차원 척도법
# 입력 : 보존 - 거리값

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def make_hello(N=1000, rseed=42):
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold',
            size=85)
    fig.savefig('hello.png')
    plt.close(fig)

    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    print("이미지차원", data.shape)

    print(data)
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    print("만든 갯수",X.shape)
    print("곱한후의 사이즈",(X * data.shape).shape)
    i, j = (X * data.shape).astype(int).T

    mask = (data[i, j] < 1)
    X = X[mask]
    print("새로운X갯수", X.shape)
    print("원래이미지의 차수 ", data.shape)
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]

X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal')

def rotate(X, angle):
  theta = np.deg2rad(angle) # 라디안
  R = [[np.cos(theta), np.sin(theta)],
       [-np.sin(theta), np.cos(theta)]]
  print(type(R))
  return np.dot(X, R) # 데이터 @ 변환행렬 -> 회전
X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal')

# HELLO -> 1000 * 1000 -> 정방행렬 대칭행렬
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
print(D.shape)
D[:5, :5]

from sklearn.manifold import MDS
# 보존 : 거리값 보존
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal')
print(out)

# 데이터를 3차원으로 확대
rng = np.random.RandomState(10)
C = rng.randn(3, 3) # 임의의 행렬
print(np.dot(C, C.T)) # 대칭행렬 만드는 방법
e, V = np.linalg.eigh(np.dot(C, C.T)) # 대칭행렬 -> 고유값 분해
print("eigenvalue", e)
print("eigenvector", V)
print(np.dot(V[0], V[1])) # 내적을 내면 직교
print(np.dot(V[:,0], V[:,1]))

# 2차원 -> 3차원으로 변경하려면 필요한 행렬
# 1000x2 2x3 정직교하는 행렬이 필요

# 차원 확대를 일반화
def random_projection(X, dimension=3, rseed=42):
  assert dimension >= X.shape[1] # 열 : 강제 예외발생
  rng = np.random.RandomState(rseed) # 임의 축으로
  C = rng.randn(dimension, dimension) # 3x3
  print("C는 ",C.shape)
  print(np.dot(C, C.T)) # 대칭행렬
  e, V = np.linalg.eigh(np.dot(C, C.T)) # 고유값 분해
  print("V는 ", V.shape)
  print("차원은 ", V[:X.shape[1]])
  return np.dot(X, V[:X.shape[1]]) # 행렬곱이 되어야 하니까
  # 1000x2 2x3 행을 앞의 행렬의 열수만큼
print(X.shape)
print(X.shape[1])
print("데이터의 차원은", X.shape)
X3 = random_projection(X, 3)
X3.shape

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2], **colorize)
ax.view_init(azim=45, elev=15)

from sklearn.manifold import MDS
model = MDS(n_components=3, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal')

# 비선형 변환
def make_hello_s_curve(X):
  t = (X[:, 0] - 2) * 0.75 * np.pi # 회전
  x = np.sin(t)
  y = X[:, 1] # 변화없음 -> 회전 중심축
  z = np.sign(t) * (np.cos(t) - 1) # 회전
  print("x값", np.ptp(x))
  print("y값", np.ptp(y))
  print("z값", np.ptp(z))
  return np.vstack((x, y, z)).T
XS = make_hello_s_curve(X)

ax = plt.axes(projection='3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)

model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal')
# MDS는 비선형변환에 취약하다 -> 거리값을 유지하지 못하고 있다

# Locally linear embedding (LLE) : 로컬위주
# graph 기반
# edge연결 : 근접 데이터 유지
# node간의 전역적 관계
from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified', eigen_solver='dense')
out = model.fit_transform(XS)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15)

from sklearn.manifold import Isomap # 전역관계를 중시(국소관계는 취약)
iso_model = Isomap(n_components=2, n_neighbors=15)
X_iso = iso_model.fit_transform(X3)
plt.figure(figsize=(7, 6))
plt.scatter(X_iso[:, 0], X_iso[:, 1], **colorize)
plt.title("Isomap Dimension Reduction", fontsize=14)
plt.axis('equal')
plt.show()

# 국소적관계와 전역적관계를 균형있게 보존하기 위해 최적화
# 비지도학습 - 그라프 기반
import umap
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=100,
    min_dist=0.3,
    random_state=42
)
X_umap = umap_model.fit_transform(X3)

plt.figure(figsize=(7, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], **colorize)
plt.title("UMAP", fontsize=14)
plt.axis('equal')
plt.show()