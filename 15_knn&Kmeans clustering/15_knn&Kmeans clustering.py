import matplotlib.pyplot as plt # 시각화 script
# !apt-get update -qq # ubuntu 설치 명령어 (물어보지 말고 설치해라)
# !apt-get install fonts-nanum* -qq # 나눔 폰트 / 시스템에서 관리
import matplotlib.font_manager as fm # 글꼴 관리
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  name='NanumGothic')
# truetype font (t/f)
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'})
plt.rcParams["axes.unicode_minus"] = False # 마이너스 기호 깨짐 보완

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9],
        [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9],
        [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])

k=5
test_datapoint = [4.3, 2.7]
plt.figure()
plt.title("입력데이터")
plt.scatter(X[:,0], X[:,1], marker = "o", s=75, color="black")

# ball_tree : 위치와 반지름으로 저장
# 고속 검색: 반경에 있는 데이터만 빠르게 검색
knn_model = NearestNeighbors(n_neighbors = k, algorithm = "ball_tree").fit(X)
# 거리값, 인덱스
distances, indices = knn_model.kneighbors([test_datapoint])
print(indices.shape) # (1,5)
for rank, index in enumerate(indices[0][:k], start = 1):
  print(str(rank) + "==>", X[index])
distances

# 시각화
plt.figure()
# x축,y축
plt.scatter(X[:, 0], X[:, 1], marker = "o", s=75, color = "k")
plt.scatter(X[indices][0][:][:,0], X[indices][0][:][:,1], marker = "o", s=250, color = "k", facecolors = "none")
plt.scatter(test_datapoint[0], test_datapoint[1], marker = "x", s=75, color = "k")
plt.show()

# KNN 분류기
# KNN 회귀
# sklearn은 모든 모델이 classifier / regressor가 존재
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris() # 시험데이터
# 전처리 - 중요, 일반
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state=42)

# 가중치: uniform : 균등권리,
# distance : 거리의 역수 가중치 부여(가까운 데이터 더 중요)
# 투표
knn_clf = KNeighborsClassifier(n_neighbors = 5, weights = "uniform")
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)
print(f"예측된 클래스: {y_pred}")
print(f"모델 정확도: {knn_clf.score(X_test, y_test):.4f}")
# model shopping

# 회귀
import joblib
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([10,12,14,16,18,20,22,24,26,28])

# 연속된 수치 - 평균
# distance : 거리값의 역수로 가중치를 부여 계산(가중평균)
knn_reg = KNeighborsRegressor(n_neighbors =3, weights = "distance")
knn_reg.fit(X,y)
X_test = np.array([[5.5],[6.5],[7.5]])
y_test = np.array([19,21,23])
y_pred = knn_reg.predict(X_test)
print(f"예측된 값: {y_pred}")

# KNeighborsTransfomer 그래프 생성과 networkx를 이용한 시각화
# deep learning 모델 표현은 graph
# 데이터를 희소행렬로 표현하고 그래프 모델에 데이터를 제공하는 전치러 역할 -> KNeighborsTransformer
# spectral clusting -> 입력: graph -> 인접 행렬로 변환 -> 라플라시안행렬
from sklearn.neighbors import KNeighborsTransformer
from sklearn.datasets import make_moons
import networkx as nx
# 선형으로 나눈기 어려운 형태의 데이터(반달 데이터)
# 종속변수가 불필요
X, _ = make_moons(n_samples = 15, noise =0.1, random_state = 42)
# 트리(단방향 성장), 그래프(양방향, 다방향)의 차이
# 하나의 node가 3개까지만 연결
knn_transformer = KNeighborsTransformer(n_neighbors =3, mode = "distance")
knn_graph = knn_transformer.fit_transform(X)


# 시각화
# 0을 희소행렬 - 표현하지 말라
edges = np.column_stack(np.where(knn_graph.toarray() >0))
# 관계분석 : 중심성 분석
G = nx.Graph()
G.add_edges_from(edges)
# node(정보) + edge(연결성)
plt.figure(figsize=(6,6))
nx.draw(G, pos = X, with_labels = True, node_color = "lightblue", edge_color = "gray", node_size=500, font_size =10)
plt.title("K-Neighbors Gragh(k=3)")
plt.show()

# isomap
# mamifold 다양체 : 차원에 따라서 같은 모양을 다양하게 볼 수있다
from sklearn.manifold import Isomap
# KNN 알고리즘
# 게임 2.5게임 - 차원축소
# 2차원 -> 1차원으로 축소
# 거리값
isomap = Isomap(n_neighbors =3, n_components =1)
X_reduced = isomap.fit_transform(X)

plt.figure(figsize=(6,4))
plt.scatter(X_reduced, np.zeros_like(X_reduced), c="blue", alpha =0.6, edgecolors = "k")
plt.title("Isomap-based Dimensionality Reduction")
plt.xlabel("Reduced Dimension")
plt.show()

# LOF 로컬 이상 모델(지역 접근성을 판단)
# 데이터로 부터 정상 거리값을 학습 (정상으로부터 벗어나면 이상치)
X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9],
              [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9],
              [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1],
              [12.0, 12.0]])

from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors =3)
outlier_scores = lof.fit_predict(X) # 2차원
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=outlier_scores, cmap = "coolwarm", edgecolors = "k", s=100)

plt.title("Local Outlier Factor(LOF) - Anomaly Detection")
plt.colorbar(label = "LOF Score(-1 = Outlier, 1 = Normal)")
plt.show()

# KNN으로 얼굴 이미지 분류
# 얼굴인식 알고리즘 테스트 이미지 - 라벨
from sklearn.datasets import fetch_lfw_people
# 사람당 최소한 20장의 이미지
# 사이즈 축소
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
print(image_shape) # (87, 65)

fig, axes = plt.subplots(2,5, figsize=(15,8), subplot_kw={"xticks": (), "yticks":()})
for target , image, ax in zip(people.target, people.images, axes.ravel()):
  ax.imshow(image)
  ax.set_title(people.target_names[target])

# people은 bunch : data, target, target_names
print(people.target[0:10], people.target_names[people.target[0:10]])
# 이미지 차수: (3023, 87, 65) : 3023장의 이미지
print("이미지 차수: {}".format(people.images.shape))
print("클래스 차수: {}".format(len(people.target_names))) # 62명

# 사람별로 몇장의 이미지가 있는지 확인
# bincount : target에 들어있는 숫자 정렬 - 인덱스 순서대로 카운트된 결과를 리턴
counts = np.bincount(people.target)
for i, (count,name) in enumerate(zip(counts, people.target_names)):
  print("{0:25} {1:3}".format(name, count), end= "   ")
  if (i + 1) % 3 == 0:
    print()

# 50장으로 이미지 제한 (true/false)
np.bool = np.bool_ #  버전이 바뀌면서 형태가 바뀜
mask = np.zeros(people.target.shape, dtype = np.bool)
for target in np.unique(people.target): # 0 ~ 61
  mask[np.where(people.target == target)[0][:50]] = 1
mask.shape

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255. # 이미지 정규화
len(y_people)
np.bincount(y_people)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify = y_people, random_state = 0)
knn = KNeighborsClassifier(n_neighbors=1) # 픽셀간 거리값
knn.fit(X_train, y_train)

print("1-최근접 이웃의 테스트 세트 점수: {:.2f}".format(knn.score(X_test, y_test))) # 0.22 -> 수치 낮음 : 안좋음

X_train.shape # (1547, 5655)

# 주성분 분석
# 분해 : 이미지는 픽셀 1개가 변수 1개 -> 5655 차원
# 100개만으로도 가능한 이유: 모든 변수가 참여하여 만든 100개임
from sklearn.decomposition import PCA
# 100개의 주성분 ----- 분산이 점점 줄어듬 -> 이걸 1로 통일시킴
# 백색화 = 주성분으로 선택된 값을 각 주성분의 표준편차로 나누어서 분산을 동일하게 만들어서 같은 중요도 학습하게 해주는것
pca = PCA(n_components = 100, whiten=True, # whiten: 백색화
          random_state = 0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("훈련차원: {}".format(X_train_pca.shape))

# 22% -> 30 %로 오름
# 5655 -> 100차원으로 차원축소했음에도 불구하고 증가함
knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(X_train_pca, y_train)
print("테스트 세트 정확도: {:.2f}".format(
    knn.score(X_test_pca, y_test)))

# 87 x 65 -> 5655 -> 공분산 행렬(5655*5655)
fig, axes = plt.subplots(3,5, figsize=(15,12),
                         subplot_kw ={"xticks": (), "yticks": ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
  ax.imshow(component.reshape(image_shape), cmap="viridis")
  ax.set_title("PCA component {}".format((i+1)))

# (1547, 100) * (1547, 5655)
# (1547, 5655)
res = X_train_pca @ pca.components_ # @ : 행렬곱
res.shape

pca.components_.shape

# 100개의 주성분축으로 재표현 이미지
image_shape = (87, 65)
fig, axes = plt.subplots(3,5, figsize=(15,12),
                         subplot_kw={"xticks": (), "yticks":()})
for i, (component, ax) in enumerate(zip(res[:15], axes.ravel())):
  ax.imshow(component.reshape(image_shape), cmap = "viridis")
  ax.set_title("PCA component {}".format((i+1)))
plt.tight_layout()
plt.show()

# kmeans
# 거리기반 비지도학습
# k개의 그룹수를 선택
# 1. k개의 중심을 초기화
# 2. k개의 중심과의 거리값을 계산
# 3. 거리가 가까운 그룹으로 배치
# 4. 중심을 재계산하고 다시 2번으로 돌아가서 과정을 반복
# 중심이 3개가 결정, 데이터는 그룹으로 할당이 됨
# 그룹의 특징을 알려면 중심분석 후처리가 필요함
from sklearn import cluster, datasets
import seaborn as sns
iris = datasets.load_iris()
X = iris.data[:, :2] # 2개변수만 사
y_iris = iris.target
km2 = cluster.KMeans(n_clusters=2).fit(X)
km3 = cluster.KMeans(n_clusters=3).fit(X)
km4 = cluster.KMeans(n_clusters=4).fit(X)

# 시각화
plt.figure(figsize =(9,3))
plt.subplot(131)
# 군집수를 늘리면 inertia_(군집간거리)가 줄어듬
plt.scatter(X[:,0], X[:,1], c=km2.labels_) # 군집번호
# 군집내에서 중심과의 거리값의 합계
plt.title("K=2, J=%.2f" % km2.inertia_) # WSS(within sum of square)

plt.subplot(132)
plt.scatter(X[:,0], X[:,1], c=km3.labels_)
plt.title("K=3, J=%.2f" % km3.inertia_)

plt.subplot(133)
plt.scatter(X[:,0], X[:,1], c=km4.labels_)
plt.title("K=4, J=%.2f" % km4.inertia_)

km4.cluster_centers_ # 군집수(4) * 변수수(2)

km3.labels_

km3.inertia_

from sklearn.metrics import silhouette_score, silhouette_samples

silhouette_scores={
    "K=2": silhouette_score(X, km2.labels_),
    "K=3": silhouette_score(X, km3.labels_),
    "K=4": silhouette_score(X, km4.labels_)
}
print(silhouette_scores) # 모델 전체의 평


silhouette_vals_2 = silhouette_samples(X, km2.labels_)
silhouette_vals_3 = silhouette_samples(X, km3.labels_)
silhouette_vals_4 = silhouette_samples(X, km4.labels_)
len(silhouette_vals_2)
fig, axes = plt.subplots(1, 3, figsize = (15, 5))
for ax, silhouette_vals, labels, title in zip(
     axes,
     [silhouette_vals_2, silhouette_vals_3, silhouette_vals_4],
     [km2.labels_, km3.labels_, km4.labels_],
     ["K = 2", "K = 3", "K = 4"],
):
    y_lower = 10
    for i in np.unique(labels):
      cluster_silhouette_vals = silhouette_vals[labels == i]
      cluster_silhouette_vals.sort()
      y_upper = y_lower + len(cluster_silhouette_vals)
      ax.fill_betweenx(
          np.arange(y_lower, y_upper), 0,
          cluster_silhouette_vals, alpha = 0.7
      )
      ax.text(0.01, y_lower +
              len(cluster_silhouette_vals) / 2, f"Cluster {i}")
      y_lower = y_upper + 10
    ax.axvline(x = silhouette_score(X, labels), color = "red",
               linestyle = "--", label = "평균 실루엣 계수")
    ax.set_xlabel("실루엣 계수")
    ax.set_ylabel("Samples")
    ax.set_title(f"실루엣 시각화 {title}")
    ax.legend()
plt.tight_layout()
plt.show()
# 모든 데이터에 대해서 실루엣계수를 계산하여 시각화
# 수식: 데이터 한개가 있다면 [가장 가까운 그룹의 요소들과의 거리(bi)- 그룹 내 요소들과의 거리(ai)]
#                                               max(ai, bi)
# 값의 범위는 -1 ~ 1사이의 값
# 1이면 적절한 그룹에 배치
# 0이면 경계선
# -1이면 다른 그룹에 잘못 배치됨

from sklearn import cluster
image = plt.imread("/content/drive/MyDrive/dataset/Lotus.jpg")
plt.figure(figsize = (5,3))
plt.imshow(image)

image.shape # (183, 275, 3) 높이 x 가로 x 컬러값

print("이미지 바이트 수", image.shape[0] * image.shape[1] * image.shape[2])

x,y,z = image.shape
# 부동소수점 이미지로 변환 / 이미지 정규화
image = np.array(image, dtype=np.float64) / 255
image_2d = image.reshape(x*y, z)
image_2d.shape
# (50325, 3)

# 군집수: 32
# 군집대상
kmeans_cluster = cluster.KMeans(n_clusters=32)
kmeans_cluster.fit(image_2d)
# 군집의 중심 32개 : 군집의 대표 컬러값
cluster_centers = kmeans_cluster.cluster_centers_
cluster_centers

len(cluster_centers)

cluster_centers.shape

# 군집번호
cluster_labels = kmeans_cluster.labels_
len(cluster_labels)
# (183, 275)
# 컬러값
# 50325 -> cluster label

plt.figure(figsize=(15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x,y,z))

# 문제 : 이미지는 부동소주점 이미지로 표현되고 있는데 이를 다시 unsigned int형으로 변환한 다음 출력하시오.
# 255로 나눠줬던걸 다시 곱해줘야함
# 데이터 타입을 변경


plt.imshow((cluster_centers[cluster_labels].reshape(x,y,z) *255).astype(np.uint8))

from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import numpy as np
n_colors = 64
china = load_sample_image("china.jpg")
print(type(china))

china = np.array(china,dtype=np.float64) / 255
w, h, d = original_shape = tuple(china.shape)
print(w,h,d)

427 * 640

from sklearn.cluster import KMeans
image_array = np.reshape(china, (w*h, d))
t0 = time()
# 273280 -> 1000장만 추출(시간단축)
image_array_sample = shuffle(image_array, random_state = 0)[:1000]
kmeans = KMeans(n_clusters = n_colors, random_state=0).fit(image_array)
print("done in %0.3fs." % (time()- t0))
# 1000 장 : 0.047s, 273280장 전부 :6.789s.

kmeans.cluster_centers_

# LABEL -> 1000개만 만들어짐 (273280은 라벨이 안만들어짐)

# 비지도학습 모델 예측
t0 = time()
labels = kmeans.predict(image_array)
print("don %0.3fs." % (time() - t0))

# 상대방 전달할 때 중심값으로 이줘진
# codebook, 라벨만 전달, 이미지 사이즈
# 중심값의 개수 * 컬러수(RGB)
def recreate_image(codebook, labels, w, h):
  d = codebook.shape[1] #3 / 코드북에서 컬러값을 찾고
  image = np.zeros((w,h,d)) # 이미지값이 들어가 공간 확보
  label_idx = 0
  for i in range(w):
    for j in range(h):
      image[i][j] = codebook[labels[label_idx]]
      label_idx += 1
  return image

# 시각화
plt.figure()
plt.clf()
ax = plt.axes([0,0,1,1])
plt.axis("off")
plt.title("Original(96,615 colors)")
plt.imshow(china)

# 전달 : 컬러북, 라벨, 이미지 사이즈 w x h => 사이즈를 줄여서 전단
plt.figure(2)
plt.clf()
ax = plt.axes([0,0,1,1])
plt.axis("off")
# 벡터 양자화
plt.title("Vector Quantized(64 colors, K-Means)")
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))


# kmeans를 이용한 원형 이상치 제거
from sklearn.datasets import make_blobs
x, label = make_blobs(100, centers =1)
X.shape

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=1)
kmeans.fit(X)

f, ax = plt.subplots(figsize=(7,5))
ax.set_title("원형이상치제거")
ax.scatter(X[:,0], X[:,1], label = "Points")
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1], label ="Centroid", color = "r")
ax.legend(loc="best")

distances = kmeans.transform(X) # 데이터를 원점으로부터 거리계산
distances
np.sort(distances) # 값정렬

# 문제 : 내림차순으로 정렬한 다음 가장 먼 5개의 인덱스를 선택하시오.
# 2차원 출력 -> 정렬은 1차원만 가능
# 인덱스 정렬 : np.argsort
sorted_idx = np.argsort(distances.ravel())[::-1][:5]
# 가장 먼 5개가 이상치가 됨

f, ax = plt.subplots(figsize=(7,5))
ax.set_title("인덱스 정렬기준 원형 이상치 탐색", fontsize=15)
ax.scatter(X[:,0], X[:,1], label ="정점")
ax.scatter(kmeans.cluster_centers_[:,0],
           kmeans.cluster_centers_[:,1],
           label = "중심", color = "r")
ax.scatter(X[sorted_idx][:, 0],
           X[sorted_idx][:,1],
           label = "이상치", edgecolors ="g",
           facecolors ="none", s= 100)
ax.legend(loc="best")

# 문제 -> 메일 전달
# distace가 2보다 크면 이상치라고 한다.
# 이를 판별해서 시각화하시오

# 내 풀이
# 거리 2 초과인 데이터의 인덱스 추출
outlier_idx = np.where(distances.ravel() > 2)[0]
f, ax = plt.subplots(figsize=(7,5))
ax.set_title("Distance > 2 이상치 탐색", fontsize=15)

# 전체 데이터
ax.scatter(
  X[:,0], X[:,1], label = "정점")

# 군집 중심
ax.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color = "r",
    label = "중심"
)

# 이상치
ax.scatter(
    X[outlier_idx][:,0],
    X[outlier_idx][:,1],
    edgecolors = "g",
    facecolors = "none",
    s= 100,
    label = "이상치"
)

ax.legend(loc="best")
plt.show()

# 선생님 풀이

distances = kmeans.transform(X)
threshold = 2 # 거리값을 주거나
threshold = np.percentile(distances, 95) # 5%만 이상치
result = distances > threshold
print(len(result))
X[result.ravel()]
len(X[result.ravel()])

# 시각화
f, ax = plt.subplots(figsize=(7,5))
ax.set_title("단일 클러스터 임계치기준 이상치 탐색", fontsize =12)
ax.scatter(X[:,0], X[:,1], label = "정점")
ax.scatter(kmeans.cluster_centers_[:,0],
           kmeans.cluster_centers_[:,1],
           label = "중심", color = "r")
ax.scatter(X[result.ravel()][:,0],
           X[result.ravel()][:,1],
           label = "이상치", edgecolors = "g",
           facecolors = "none", s=100)
ax.legend(loc="best")

# hierachical clustering 계층적 군집화
# 하위에서 상위로 군집화
# 모델용으로 사용하지 않고 k값을 결정용으로 사용

# 연결법
# 최단연결법
# 최장연결법
# 평균연결법
# ward법 분산의 제곱량이 증가 하는 방향을 모든 점을 고려해서 구함

# ward법
# 정답이 있을 때의 평가 방법
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.datasets import make_blobs

X,y = make_blobs(random_state=0, n_samples =12)
linkage_array = ward(X) # 군집간 거리값을 구함
dendrogram(linkage_array)

ax = plt.gca() # 현재 축을 획득
bounds = ax.get_xbound() # x축 시작점 ~ 끝점을 구함

ax.plot(bounds, [7.25, 7.25], "--", c="k") # 2개의 군집분할선
ax.plot(bounds, [4,4], "--", c="k")  # 3개의 군집분할선
ax.text(bounds[1], 7.25, "두 개 클러스터", va = "center", fontdict={"size":15})
ax.text(bounds[1], 4, "세 개 클러스터", va = "center", fontdict={"size":15})
plt.xlabel("샘플 번호")
plt.ylabel("클러스터 거리")

# !pip install mglearn

# DBSCAN
# k값이 불필요
# 밀도기반
# epsilon거리, minumum point 최소그룹 요소수
# 거리를 기준으로 할 때는 정규화 하는 것이 유리함

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import mglearn

X,y = make_moons(n_samples=200, noise = 0.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
# eps 작아지면 군집수는 -> 늘어남
# min_samples 커지면 -> 군집수는 줄어듬
# 이 관계를 시각화 한것 : mglearn
X_scaled = scaler.transform(X)
dbscan = DBSCAN(eps = 0.5, min_samples= 10) # k값 없음
# 군집이 안되면 -> outlier 이상치
clusters = dbscan.fit_predict(X_scaled)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, s =60, edgecolors = "black")
plt.xlabel("특성 0")
plt.ylabel("특성 1")

mglearn.plots.plot_dbscan()