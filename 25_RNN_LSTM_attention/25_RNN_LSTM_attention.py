# 시계열 데이터
import pandas
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy
numpy.random.seed(7)
dataframe = pandas.read_csv(
    "/content/drive/MyDrive/dataset/Passengers.csv",
    usecols = [1], engine = "python", skipfooter = 3)
dataset = dataframe.values
dataset = dataset.astype("float32")
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0: train_size, :], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# 시계열 데이터 -> 열이 한개 -> 데이터를 재구성 해줘야함
# 데이터 재구성 함수 : create_dataset
def create_dataset(dataset, look_back = 1):
  dataX, dataY = [], [] # 독립변수, 종속변수 선언
  for i in range(len(dataset)- look_back -1): # 맨마지막은 처리안됨
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i+look_back, 0])
  return numpy.array(dataX), numpy.array(dataY)

look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
model = Sequential()
model.add(Dense(8, input_dim = look_back, activation = "relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer = "adam")
model.fit(trainX, trainY, epochs = 200, batch_size = 2, verbose = 2)
testScore = model.evaluate(testX,testY, verbose = 0)
print("Test Score: %.2f MSE(%.2f RMSE)" % (testScore, math.sqrt(testScore)))

# 시각화
import matplotlib.pyplot as plt
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back*2)+1:len(dataset)-1, :] = testPredict
plt.plot(dataset)
plt.show()
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# 뭘또 바꾼담
# 위에걸 TimeseriesGenerator 요걸로 사용해보는 듯
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])
print(data[:10])
data_gen = TimeseriesGenerator(data, targets,
                               length =10, # 데이터 윈도우 사이즈
                               sampling_rate = 2, # 2개씩 건너띄면서
                               batch_size=2)
assert len(data_gen) == 20
batch_0 = data_gen[0]
print("데이터 구조", batch_0)
# 3차원 구조 LSTM에 데이터 입력 시 (배치사이즈, 타임스텝(셀 수), 특성(하나의 셀에 입력되는 특성))
# 이미지는 4차원 배치사이즈 = 배치사이즈, 행, 열 , channel

look_back = 2
batch_size =2
train_generator = TimeseriesGenerator(
    data = train,
    targets = train,
    length = look_back,
    batch_size = batch_size
)
test_generator = TimeseriesGenerator(
    data= test,
    targets = test,
    length = look_back,
    batch_size = batch_size
)

# from tensorflow.keras.layers import Dense, LSTM - > 쌤 코드 구식이라 밑줄 표시남
from keras.layers import Dense, LSTM # 요즘 방식
model = Sequential(name = "TimeseriesGenerator_LSTM")
# 나가는 차수 : hidden 은닉 차수
# time step : 2 -> cell이 2개가 만들어짐
# cell이 몇개인가라는 것과 동일
# 특징이 : 1
model.add(LSTM(8, input_shape=(look_back, 1), activation = "relu"))
model.add(Dense(1))
model.compile(loss = "mean_squared_error", optimizer = "adam")

model.fit(train_generator, epochs = 200, verbose = 2)
testScore = model.evaluate(test_generator, verbose = 0)
print("Test Score: %.2f MSE (%.2f RMSE)" % (testScore, math.sqrt(testScore)))

# 임베딩
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts =[
    "나는 자연어 처리를 좋아해",
    "텍스트를 숫자로 변환해보다",
    "딥러닝 모델은 고정된 입력을 필요로 해"
]
tokenizer = Tokenizer(oov_token = "<00V>") # out of token : 단어 token으로 분석 : 없는 단어인 경우
tokenizer.fit_on_texts(texts) # 단어장을 생성
sequences = tokenizer.texts_to_sequences(texts) # 텍스트를 숫자로 변환
maxlen = 8
# 같은 사이즈의 텍스트 : 행렬곱
padded_sequences = pad_sequences(sequences, maxlen = maxlen,
                                 padding = "post", truncating = "post")

padded_sequences

# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer

# 모델 로드
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# 문장을 숫자 벡터(Embedding)로 변환하는 사전학습된 Sentence Transformer 모델

# 문장
sentences = [
    "나는 자연어 처리를 좋아해",
    "텍스트를 숫자로 변환해보다",
    "딥러닝 모델은 고정된 입력을 필요로 해"
]

# 문장단위로 임베딩 생성 (3,384)
sentence_embeddings = sbert_model.encode(sentences)

# 결과 확인
print("문장 임베딩 shape:", sentence_embeddings.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (384,)),
    # 384 x 128
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.2),
    # 128 x 1
    tf.keras.layers.Dense(1, activation = "sigmoid")
])
labels = np.array([1,0,1]) # 긍정,부정
model.compile(optimizer = "adam", loss = "binary_crossentropy",
              metrics = ["accuracy"])
model.fit(sentence_embeddings, labels, epochs = 10)

# # 위에거 그대로 LSTM으로 해봄 -> 참고만 실행 nope

# # (3, 384) -> (3, 1, 384)
# sentence_embeddings_lstm = sentence_embeddings.reshape(-1, 1, 384)

# model = tf.keras.Sequential([
#     # LSTM 입력: (타임스텝수, 특성수)
#     tf.keras.layers.Input(shape=(1, 384)),
#     tf.keras.layers.LSTM(64, activation="tanh"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])

# labels = np.array([1, 0, 1])

# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.fit(sentence_embeddings_lstm, labels, epochs=10)

new_sentences = [
    "인공지능 머신러닝 알고ㅓ리즘을 개발하는 중입니다.",
    "오늘 점심에 맛있는 음식을 먹으러 갈까요?"
]

new_embeddings = sbert_model.encode(new_sentences)
predictions = model.predict(new_embeddings, verbose = 0)

for sentences, prob in zip(new_sentences, predictions):
  pred_class = 1 if prob >= 0.5 else 0
  print(f"-> 문장: '{sentences}")
  print(f"긍정(1) 클래스 확률: {prob[0]: .4f} | 최종 예축결과: Class {pred_class}")

# 문장 유사도 구하
from sklearn.metrics.pairwise import cosine_similarity
query_sentence = "기계학습 모델에 텍스트 데이터를 집어넣으려고 해요."
query_embedding = sbert_model.encode([query_sentence])
similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
print(f"-> 검색어(Query): '{query_sentence}'")
for i, score in enumerate(similarities):
  print(f" 비교 문장 {i+1}: '{sentences[i]}' -> 의미적 유사도 점수: {score: .4f}")

# 뉴스분류
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
# 사이즈 계산
np.random.seed(1337)
max_words = 1000 # 뉴스기사의 길이가 상이
batch_size = 100
nb_epoch = 200
# train_test_split
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = max_words, test_split = 0.2)
# 0,1,2,3 경제, 4: 사회
nb_classes = int(np.max(y_train))+1 # 분류개수
tokenizer = Tokenizer(nb_words = max_words)

unique_classes = np.unique(y_train) # 기사종류
nb_classes_unique = len(unique_classes)

# 모델 생성
nb_classes = int(46)
X_train = tokenizer.sequences_to_matrix(X_train, mode = "binary")
X_test = tokenizer.sequences_to_matrix(X_test, mode = "binary")
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape = (max_words,), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer ="adam", metrics = ["accuracy"])
history = model.fit(X_train, Y_train, epochs = nb_epoch, batch_size = batch_size,
                    verbose = 0, validation_split = 0.1)
score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 1)
print("Model accuracy: %.2f%%" % (score[1]*100))
print("Model loss: %.2f%%" % (score[0]*100))

X_train[0] # 인코딩 되어있음(숫자로 나)

# 임베딩값 == 단어
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = max_words, test_split=0.2)
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text_indices):
  decoded_words = [reverse_word_index.get(i-3, "?") for i in text_indices]
  return "".join(decoded_words)
first_sample_indices = X_train[0]
decoded_text = decode_review(first_sample_indices)
decoded_text

# attention
import tensorflow as tf
x = [ [1,0,1,0], [0,2,0,2], [1,1,1,1]] # 3x4
x = tf.convert_to_tensor(x, dtype=tf.float32)
print(x)

# 키이(key), 쿼리(query), 밸류(value)
# attention -> 이미지, 사운드, 텍스트, 동영상
# 이미지 (CNN: local 공간 특징) + Attention(전역적 특징: global한 특징)
# text에서 개발된 알고리즘
# ViT(vision transformer) : encoding만
# 데이터입력 3개의 가중치 작동(key가중치, 쿼리가중치, 밸류가중치)
# key(질의에 대응하는 값), query(질의), value(표현하고자 하는 값)
# 입력데이터 -> 자동으로 key, query, value로 나눠짐
# cnn : 공간적 지역특성, rnn : 순서가 있는 특징 , attention : 관계 특성
# self attention은 순서를 보장하지 않음 -> 스스로의 관계속에서 특징을 추출함
# self attention + position encoding(순서를 보장함) -> transformer를 설계
# transformer = encoding(특성추출) + decodint(생성)
# transformer는 태생이 번역망
# encoding만 빼서 : BERT를 만듬            => 분류, 감정분석 잘함
# decoding만 빼서 : GPT -> chatgpt를 만듬  => 언어생성 잘함

w_key= [
    [0,0,1],
    [1,1,0],
    [0,1,0],
    [1,1,0]
]

w_query = [
    [1,0,1],
    [1,0,0],
    [0,0,0],
    [0,1,1]
]

w_value = [
    [0,2,0],
    [0,3,0],
    [1,0,3],
    [1,1,0]
]

w_key = tf.convert_to_tensor(w_key, dtype = tf.float32)
w_query = tf.convert_to_tensor(w_query, dtype = tf.float32)
w_value = tf.convert_to_tensor(w_value, dtype = tf.float32)
# key, value, query => 가중치 학습
# FFNN : 가중치 초기화, CNN.filter가 초기화
# RNN cell 가중치 초기화
# attention : key가중치, value가중치, query가중치

keys = tf.matmul(x, w_key)
querys = tf.matmul(x, w_query)
values = tf.matmul(x, w_value)
print(keys)
print(querys)
print(values)
# 너 나한테 중요한지? 확인

attn_scores = tf.matmul(querys, keys, transpose_b = True)
print(attn_scores)

# 확률값으로 변환
attn_scores_softmax = tf.nn.softmax(attn_scores, axis =-1)
print(attn_scores_softmax)


attn_scores_softmax = [
    [0.0, 0.5, 0.5],
    [0.0, 1.0, 0.0],
    [0.0, 0.9, 0.1]
]
attn_scores_softmax = tf.convert_to_tensor(attn_scores_softmax)
print(attn_scores_softmax)

values[:, None]

tf.transpose(attn_scores_softmax)[:,:,None]

weighted_values = values[:, None] * tf.transpose(attn_scores_softmax)[:,:,None]
print(weighted_values)

outputs = tf.reduce_sum(weighted_values, axis =0)
print(outputs)

# keras에 attention을 적용
# lstm 배치사이즈(32), 타임스텝수(8), 특성수(8)
# 셀이 8개가 구성
inputs = tf.random.normal([32, 10, 8])
lstm = tf.keras.layers.LSTM(4) # 나가는 차수
output = lstm(inputs)
# LSTM 오른쪽으로, 위쪽으로 출력하는 값을 다 출력하시오
# hidden state - 8개의 셀의 영향을 다 고려해서 추출된 특징
# cell state - 장기기억을 유지하는 값
lstm = tf.keras.layers.LSTM(4, return_sequences = True, return_state = True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)

# keras -> transfomer네트 구성
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# query / value 만을 활용
def build_attention_transformer_model():
    query_input = tf.keras.Input(shape=(None,), dtype="int32")
    value_input = tf.keras.Input(shape=(None,), dtype="int32")
    # 차원축소
    token_embedding = tf.keras.layers.Embedding(input_dim = 1000, output_dim = 64)
    query_embeddings = token_embedding(query_input)
    value_embeddings = token_embedding(value_input)
    cnn_layer = tf.keras.layers.Conv1D( # 주변값을 고려해서 특징 추출
        filters = 100,
        kernel_size = 4,
        padding = "same"
    )
    query_seq_encoding = cnn_layer(query_embeddings) # 특징추출 결과
    value_seq_encoding = cnn_layer(value_embeddings)

    # 관계계산 : 점수 매기기
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])
    # 차원압축
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    input_layer = tf.keras.layers.Concatenate()( # 융합
        [query_encoding, query_value_attention])
    x = layers.Dense(64, activation = "relu")(input_layer)
    x = layers.Dropout(0.3)(x)
    output_layer = layers.Dense(1, activation = "sigmoid", name = "Probability")(x)
    # attention망을 감정분석에 적용
    model = Model(inputs = [query_input, value_input], outputs = output_layer, name = "Attention_Network")

    return model

# instance
model  = build_attention_transformer_model()
model.compile(optimizer = "adam", loss = "binary_crossentropy",
              metrics = ["accuracy"])
model.summary()

# key === value 일치해야 함 => 상수 * [차원이 달라도 곱셈이 ]
# query != value는 불일치해도 됨
dummy_query = np.random.randint(1, 1000, size = (32, 10))
dummy_value = np.random.randint(1, 1000, size = (32, 15))
dummy_labels = np.random.randint(0,2, size=(32, 1))
history = model.fit([dummy_query, dummy_value],
                    dummy_labels, epochs = 3, batch_size = 4, verbose = 1)

import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!',
        'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
own_embedding_vocab_size = 10

encoded_docs_oe = [one_hot(d, own_embedding_vocab_size) for d in docs]
print(encoded_docs_oe)

maxlen = 5
padded_docs_oe = pad_sequences(encoded_docs_oe, maxlen = maxlen, padding = "post")
print(padded_docs_oe)

model = Sequential()
model.add(Embedding(input_dim = own_embedding_vocab_size, output_dim = 32,
                    input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["acc"])
print(model.summary())
model.fit(padded_docs_oe, np.array(labels), epochs = 30, verbose = 0)

def predict_sentiment(sentence):
  encoded = one_hot(sentence, own_embedding_vocab_size)
  padded = pad_sequences([encoded], maxlen = maxlen, padding = "post")
  prediction = model.predict(padded, verbose = 0)[0][0]
  print(encoded)
  print(f"\n[입력] {sentence}")
  print(f"[예측확률] {prediction:.4f}")
  print(f"[예측결과] {'긍정' if prediction >= 0.5 else '부정'}")

predict_sentiment("Nice work!")
predict_sentiment("Excellent effort")
predict_sentiment("Cloud be better")

# !wget --no-check-certificate \
#     https://raw.githubusercontent.com/computationalcore/introduction-to-opencv/master/assets/test.jpg   -O test.jpg
# !wget --no-check-certificate \
#     https://raw.githubusercontent.com/computationalcore/introduction-to-opencv/master/assets/haarcascade_frontalface_default.xml   -O haarcascade_frontalface_default.xml
# !wget --no-check-certificate \
#     https://raw.githubusercontent.com/computationalcore/introduction-to-opencv/master/assets/haarcascade_smile.xml  -O haarcascade_smile.xml
# !wget --no-check-certificate \
#     https://raw.githubusercontent.com/computationalcore/introduction-to-opencv/master/assets/haarcascade_eye.xml  -O haarcascade_eye.xml
# !wget --no-check-certificate \
#     https://raw.githubusercontent.com/computationalcore/introduction-to-opencv/master/utils/common.py   -O common.py

import common
import pylab
pylab.rcParams["figure.figsize"] = (10.0, 8.0)


import cv2
import matplotlib.pyplot as plt
base_image = cv2.imread("/content/drive/MyDrive/dataset/family.jpg")
grey = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))

# 얼굴만
face_cascade = cv2.CascadeClassifier("/content/haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(grey, 1.3, 5)
# 그릴때는 좌상단 점, 우하단 점
for (x, y, w, h) in faces:
  cv2.rectangle(base_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))

# 웃는 얼굴
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
smiles = smile_cascade.detectMultiScale(grey, 1.3, 20)

for (x, y, w, h) in smiles:
    cv2.rectangle(base_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))


# 눈
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
eyes = eye_cascade.detectMultiScale(grey, 1.3, 20) # snsrjacnf

# 검출된 눈에 사각형 그리기
for (x, y, w, h) in smiles:
    cv2.rectangle(base_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)) # 결과

test_image = cv2.imread("/content/drive/MyDrive/dataset/family.jpg")
for (x,y,w,h) in faces:
  cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,0,0),2)
  for (x_s,y_s,w_s,h_s) in eyes:
    if( (x <= x_s) and (y <= y_s) and ( x+w >= x_s+w_s) and ( y+h >= y_s+h_s)):
      cv2.rectangle(test_image, (x_s,y_s),(x_s+w_s,y_s+h_s),(255,255,255),2)
  for (x_s,y_s,w_s,h_s) in smiles:
    if( (x <= x_s) and (y <= y_s) and ( x+w >= x_s+w_s) and ( y+h >= y_s+h_s)):
      cv2.rectangle(test_image, (x_s,y_s),(x_s+w_s,y_s+h_s),(0,255,0),2)
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

# 동영상 다운

# 영상다운 - 유튜브
# !pip install yt-dlp
# !pip install pafy

# !yt-dlp "https://www.youtube.com/watch?v=OMBjyeSJLqY" -o "dog.%(ext)s"

# !pip install ffmpeg-python

# !ffmpeg -ss 00:00:30 -i dog.webm -t 00:00:05 -c:v copy test2.mp4

# !ls # 파일 확인

from IPython.display import HTML
from base64 import b64encode

mp4 = open("test2.mp4", "rb").read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=500 controls>
  <source src="%s" type="video/mp4">
</video>
""" % data_url)

# 폴더생성
import os
os.mkdir("output_image")

# !ffmpeg -i test2.mp4 -r 1 ./output_image/out%d.png

# # 저장확인
# !ls output_image

import glob
filepaths = list(glob.glob("./output_image/*.png"))
len(filepaths)

os.chdir("./output_image")
path = "."
mean_height = 0
mean_width = 0
num_of_images = len(os.listdir('.'))
num_of_images

# 평균 width, height 계산
from PIL import Image
import os

for file in os.listdir("."):
    if file.endswith(".jpg") or file.endswith(".png"):
        im = Image.open(file)
        width, height = im.size
        mean_width += width
        mean_height += height

mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)

print("평균 width:", mean_width)
print("평균 height:", mean_height)

# 이미지 크기 일치시키기
from PIL import Image
import os

for file in os.listdir("."):
    if file.endswith(".jpg") or file.endswith(".png"):
        im = Image.open(file)
        imResize = im.resize((mean_width, mean_height))
        imResize.save(file, "PNG", quality=95)

print("이미지 크기 통일 완료")

# 동영상 생성
def generate_video():

    image_folder = "."
    images = [
        img for img in os.listdir(image_folder)
        if img.endswith(".jpg")
        or img.endswith(".jpeg")
        or img.endswith(".png")
    ]

    images.sort()  # 순서 정렬
    video_name = "/content/mygeneratedvideo.mp4"
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 5, (width, height))
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)
    video.release()
generate_video()

# import cv2
# from google.colab.patches import cv2_imshow
# video_path = '/content/mygeneratedvideo.mp4'
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error:파일 오픈 에러.")
# else:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             cv2_imshow(frame)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()

from google.colab.patches import cv2_imshow
import cv2

cap = cv2.VideoCapture("/content/mygeneratedvideo.mp4")

ret, frame = cap.read()

cv2_imshow(frame)

cap.release()

# 1) H.264로 다시 인코딩
# !ffmpeg -y -i /content/mygeneratedvideo.mp4 -c:v libx264 -pix_fmt yuv420p /content/mygeneratedvideo_h264.mp4

# # 2) 파일 생성 확인
# !ls -lh /content/mygeneratedvideo_h264.mp4

# 3) HTML로 다시 재생
from IPython.display import HTML
from base64 import b64encode

mp4 = open("/content/mygeneratedvideo_h264.mp4", "rb").read()

data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video width=600 controls>
<source src="%s" type="video/mp4">
</video>
""" % data_url)

# 문제: 동영상을 작성하는 과정에서 다음 기능을 추가 적용하시오.
# 300장까지는 가우시안 필터 적용
# 600장까지는 transformation rotate 15도 적용
# 600장 이후는 perspective 변환을 적용해서 동영상을 다시 작성하시오.
# 아니면 1/3씩 구간마다 ????? 못적음 -> 그냥 참고였음

# # 문제 결과인듯
# def generate_video():
#   image_folder = '.'
#   images = [img for img in os.listdir(image_folder)
#                 if img.endswith(".jpg") or
#                   img.endswith(".jpeg") or
#                   img.endswith("png")]
#   video_name = '/content/moviemaker/mygeneratedvideo.mp4'
#   os.chdir("/content/moviemaker/roses")
#   frame = cv2.imread(os.path.join(image_folder, images[0]))
#   height, width, layers = frame.shape
#   fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#   video = cv2.VideoWriter(video_name, fourcc, 5, (width, height))
#   i = 0
#   for image in images:
#         workimage=cv2.imread(os.path.join(image_folder, image))
#         rows,cols,_ = workimage.shape
#         i+=1
#         # 변환
#         if i<=  300 :
#           dst = cv2.GaussianBlur(workimage,(5,5),0)
#         elif 300<i<=600:
#           M   = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),45,1)
#           dst = cv2.warpAffine(workimage,M,(cols,rows))
#         elif 600<i :
#           pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
#           pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
#           M    = cv2.getPerspectiveTransform(pts1,pts2)
#           dst  = cv2.warpPerspective(workimage,M,(cols,rows))
#         video.write(dst)
#   cv2.destroyAllWindows()
#   video.release()

# 과제
# RNN으로 시계열 모델 구성하기
# 시계열 데이터를 검색 다운로드해서 예측하는 모델을 작성하시오.
