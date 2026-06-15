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

plt.rcParams['axes.unicode_minus'] = False

# one-hot-encoding
# 데이터
# (숫자, 문자, 이미지(파일로딩 -> 숫자화),
# 텍스트(encoding A: 64, a: 97),
# 동영상(실시간 임베딩 (장치)), 사운드)
# embedding : 텍스트 -> vector
import numpy as np
sentences = [
    "인공지능 모델 개발",
    "딥러닝 모델 학습"
]
word_set = set() # 키이값 - 중복을 허용하지 않음
for sentence in sentences: # 인공지능 모델 개발
  for word in sentence.split():
    word_set.add(word) # 중복은 제거 (unique한 단어)
# 단어 -> 숫자
word_to_index = {word: idx for idx, word in enumerate(
    sorted(list(word_set)) # 가나다
)}
# 모델 결과
# 숫자 -> 단어
index_to_word = {idx: word for word, idx in word_to_index.items()}

VOCAB_SIZE = len(word_to_index)

word_to_index

index_to_word

# 임베딩 -> vector화
target_sentence = "딥러닝 모델 학습"
tokens = target_sentence.split()
one_hot_vectors = []
#             '개발' '딥러닝' '모델' '인공지능' '학습'
#   딥러닝        0       1       0         0       0
#   모델          0       0       1         0       0
#   학습          0       0       0         0       1
for token in tokens:
  one_hot = np.zeros(VOCAB_SIZE) # 전체를 일단 0
  word_idx = word_to_index[token] # 몇 번째 인덱스
  one_hot[word_idx] = 1.0 # 해당 인덱스를 1
  one_hot_vectors.append(one_hot)
one_hot_matrix = np.array(one_hot_vectors)
print(f"3. ['{target_sentence}'] 문장의 원-핫 인코딩 행렬:")
print(one_hot_matrix)
print(f"  (행렬 크기: {one_hot_matrix.shape} -> [단어개수 x 사전크기])\n")

recovered_words_A = []
for row in one_hot_matrix: # [0, 1, 0, 0, 0]
  idx = np.where(row == 1.0)[0][0] # 결과값을 tuple로 전달
  recovered_words_A.append(index_to_word[idx])

" ".join(recovered_words_A)

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vectorizer = CountVectorizer()
# 자동으로 단어 추출 -> 정렬 -> 단어장
count_matrix = count_vectorizer.fit_transform(sentences)
# 어휘
vocab = count_vectorizer.get_feature_names_out()
print("f. 토큰화된 고유 단어 목록 (사전 순 정렬):")
print(" ", vocab.tolist())
print(f"  (단어-인덱스 매핑: {count_vectorizer.vocabulary_})\n")
# sentences = [
#     "인공지능 모델 개발",
#     "딥러닝 모델 학습"
# ]

# document-term matrix (등장 단어의 개수)
print(count_matrix.toarray())

# 중요도 때문에 가중치가 줄어든 모양으로 표현된 tfidf
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
print(np.round(tfidf_matrix.toarray(), 4))

import nltk
nltk.download('punkt_tab') # 다국어 tokenizer 툴
para = "Hello everyone."
from nltk.tokenize import sent_tokenize
sent_tokenize(para)

para_kor = "안녕하세요. 여러분. 만나서 반갑습니다."
sent_tokenize(para_kor) # 문장 파싱

from nltk.tokenize import word_tokenize
word_tokenize(para)

word_tokenize(para_kor)

# 정규표현식
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[\w']+") # 단어
tokenizer.tokenize("Can't is a contraction.")

nltk.download('stopwords') # 불용어 (의미없는 단어)

# 불용어는 list에 담아서 비교 후 제거
from nltk.corpus import stopwords
english_stops = set(stopwords.words('english')) # 중복제거
words = ["Can't", "is", "a", "contraction", "english"]
[word for word in words if word not in english_stops]

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download('averaged_perceptron_tagger_eng')
sentence = "James is working at Disney in London"
tokens = pos_tag(word_tokenize(sentence))
print(sentence)

tokens = word_tokenize(sentence)
tokens

tagged_tokens = pos_tag(tokens) # 품사 태깅
tagged_tokens # 고유명사 인식

# pip install svgling

# 고유명사에 대한 이해
ner_tree = ne_chunk(tagged_tokens) # NER 개체명인식
ner_tree

# movie_review 분류모델

nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)

documents[1][1] # 단어, 카테고리, 묶어주는 것이 필드 아이디

len(documents)

documents[100][0][:10]

# 도수 카운트
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
list(all_words)

all_words.most_common(5)

# 모든 데이터 중에 20개만 가지고 판단
word_features = list(all_words)[:20]
word_features

def document_features(document): # 리뷰 한 개 입력
  document_words = set(document) # 단어 중복 제거
  features = {}
  for word in word_features: # 20개의 단어 중 문에 있는 것
    features['contains({})'.format(word)] = (
        word in document_words
    )
  return features # 특징 추출

# NaiveBayesClassifier는 텍스트 데이터에만 적용하는 모델
# 한 행에 들어 있는 데이터 : 리뷰 + 감정분석 레이블 (pos, neg)
# 0: 리뷰, c: pos, neg
# 2000개
featuresets = [(document_features(d), c) for (d, c) in documents]
# 1900(훈련) : 100(테스트)
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

# gensim : dictionary

# !pip install gensim

from gensim import corpora # 말뭉치
texts = [
    ["apple", "orange", "banana"],
    ["apple", "apple", "orange"],
    ["computer", "science", "ai"]
]
dictionary = corpora.Dictionary(texts) # 사전작성
print(dictionary.token2id) # 텍스트 -> 숫자 매핑
print(len(dictionary))

documents = [
    "apple orange banana",
    "computer science ai"
]
texts = [doc.split() for doc in documents]
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)

dictionary = corpora.Dictionary() # 인스턴스만
dictionary.add_documents([
    ["apple", "orange"],
    ["banana"]
])
dictionary.add_documents([
    ["computer", "ai"]
])
print(dictionary.token2id)

from gensim.corpora import Dictionary
texts = [['human', 'interface', 'computer']]
dct = Dictionary(texts)
dct.add_documents([["cat", "say", "meow"], ["dog"]])
print(dct.token2id)
# bag of word
# 문서를 단어장에 있는 단어로 표현
# 순서없이 단어로만 표현
# 없는 단어는 별도로 출력
dct.doc2bow(["dog", "computer", "non_existent_word", "dog"], return_missing=True)

print("문서 수", dct.num_docs)
print("처리된 워드 수", dct.num_pos)
print("non-zeros가 아닌 것", dct.num_nnz)
print("단어 수", len(dct))

# 없는 단어는 -1로 표현
print(dct.doc2idx(["interface", "human", "not_in_dictionary"]))

# 사전에서 필터링 (제거)
dct.filter_tokens(bad_ids=[dct.token2id['cat']])
print('ema' in dct.token2id) # 포함연산자
len(dct)

import pprint
texts = [['한국', '대한', '만세']]
dct = Dictionary(texts)
dct.add_documents([["고양이", "말해", "사랑"], ["개"]])
print(dct.token2id)
print(dct[1])
dct.doc2bow(["개", "non_existent_word", "대한"], return_missing=True)

import pprint
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# stopword(불용어) 제거
stoplist = set('for a of the and to in'.split(' '))
texts = [[word for word in document.lower().split()
            if word not in stoplist]
         for document in text_corpus]

# 한번 이상 등장한 단어만 단어장에 포함
from collections import defaultdict
frequency = defaultdict(int) # 사전형
# 도수분포표를 작성
for text in texts:
  for token in text:
    frequency[token] += 1 # dict형은 임의 추가가 가능
# token에 대한 도수 확인
processed_corpus = [[token for token in text
                     if frequency[token] > 1]
                     for text in texts]
pprint.pprint(processed_corpus)

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

pprint.pprint(dictionary.token2id)

token_dict=dictionary.token2id
token_dict['interface'] # token -> id

dictionary[10] # id -> token

new_doc = 'Human computer interaction Human'
new_vec = dictionary.doc2bow(new_doc.lower().split())
new_vec

# 전체 데이터를 corpus구성
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus] # 불용어, 1개 이상 등장 단어
pprint.pprint(bow_corpus) # 9개의 문서로 구성

# 문서 중요도 고려
from gensim import models
tfidf = models.TfidfModel(bow_corpus)
tfidf

tfidf.idfs # 단어에 대한

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])

# 문서를 tfdif값으로 표현된 값
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
  print(doc)

new_doc = "graph trees"# 새 문서
new_bow = dictionary.doc2bow(new_doc.split()) # 단어 도수
new_tfidf = tfidf[new_bow] # 단어 중요도를 고려한 값으로 표현
print(new_tfidf)

# topic 분석 : tf, tfidf(단어 중요도) 값으로 분석
# topic 값을 단어로 표현하기 위해서 id2word값을 지정
# 내부적으로는 SVD에 의한 잠재변수도출 -> topic
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi_model[corpus_tfidf]

lsi_model.print_topics(2)
# 토픽명명식

# 입력된 데이터하고 잠재변수로 표현된 데이터 비교
for doc, as_text in zip(corpus_lsi, text_corpus):
  print(doc, as_text)

from gensim import similarities
# index = similarities.MatrixSimilarity(list(lsi[corpus_tfidf]))
# print(index)

sentences = [list(s) for s in movie_reviews.sents()]
print("0번 문장", sentences[0])
print("1번 문장", sentences[1])

from gensim.models.word2vec import Word2Vec
# embeding -> 단어를 vector로 구성
model = Word2Vec(sentences) # 의미있는 단어들의 벡터
model_name = 'test1'
model.save(model_name)

# word vector
print(model.wv.similarity('actor', 'actress')) # 0.86379325
print(model.wv.similarity('actor', 'she')) # 0.21320459
print(model.wv.most_similar("accident"))

model.wv.most_similar("man")

print(model.wv.most_similar(positive=['actor', 'she'], negative='actress', topn=3))

model.wv.doesnt_match('man woman child kitchen'.split())

# word2vec 문제 : bigram 미지원
# 두 가지 단어가 밀접하게 연결된 의미로 사용이 되는 경우
from gensim.models.phrases import Phrases, Phraser

# 1️⃣ 원시 말뭉치 데이터 (띄어쓰기 기반 유니그램 상태)
raw_sentences = [
    ["new", "york", "is", "a", "great", "city"],
    ["i", "love", "visiting", "new", "york"],
    ["artificial", "intelligence", "is", "changing", "the", "world"],
    ["he", "studies", "artificial", "intelligence", "at", "stanford"]
]

# 적어도 2개 이상, threshold 결합 정도
b_phrases = Phrases(raw_sentences, min_count=1, threshold=2)
bigram_detector = Phraser(b_phrases) # 속도 향상 객체
bigram_sentences = list(bigram_detector[raw_sentences])
print(bigram_sentences[0])

# one-hot-encoding 100만 단어
# 100만개 중에 1개만 1이 됨
# 원핫인코딩을 한 다음 100만 x 100의 가중치를 만들어서 차원축소
model = Word2Vec(sentences=bigram_sentences, vector_size=100, window=5, min_count=1, workers=4)

model.wv.key_to_index

# !pip install konlpy

# 꼬꼬마
from konlpy.tag import Kkma
from konlpy.utils import pprint
kkma = Kkma()
pprint(kkma.sentences(u'네. 안녕하세요. 반갑습니다.'))
# 형태소 분석

# 명사만 추출
pprint(kkma.nouns(u"아버지가 방에 들어가신다. 그 방에는 내 동생이 있다."))

from konlpy.tag import Okt
okt = Okt()
print(okt.morphs(u'단독입찰보다 복수입찰의 경우'))
print(okt.nouns(u'유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는'))

import pandas as pd
voc = "퀄컴은 미국 시애틀에서 열린 마이크로소프트 빌드 2023에서 스냅드래곤 컴퓨팅 플랫폼 기반 생성형 AI실행 등 온디바이스 AI 혁신 기술을 선보였으며, 스냅드래곤이 탑재된 윈도 11 PC용 애플리케이션 개발에 새로운 가능성을 제시했다고 25일 발표했다."
okt_pos = Okt().pos(voc, norm=True, stem=True)
print(okt_pos)

# 원하는 품사만 추출 : 명사, 형용사, 동사만 추출
okt_filtering = [x for x, y in okt_pos if y in [
    'Noun', "Adjective", "Verb"
]]
print(okt_filtering)
print(len(okt_filtering))
okt_len = pd.Series(okt_filtering)
okt_len=okt_len.apply(len)
df=pd.DataFrame({"word": okt_filtering, "len": okt_len})
df[df["len"]>1]

# !apt-get -qq -y install fonts-nanum > /dev/null
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

import requests 
from bs4 import BeautifulSoup 
from konlpy.tag import Twitter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# html 파일 - 웹브라우저 없이 로딩
res = requests.get('https://sports.v.daum.net/v/20181227140914322')
# 파싱
soup = BeautifulSoup(res.content, 'html.parser')
# 검색 : class
body = soup.select('.article_view')[0]
# paragraph로 표현되는 것을 검색
# <p></p> 안에 있는 텍스트를 가지고 옴
text = " ".join(p.get_text() for p in body.find_all('p')) 
t = Twitter() # class 인스턴스
tokens_ko = t.nouns(text) # 명사만 추출
ko = nltk.Text(tokens_ko, name = '스포츠') # 명사만 Text 객체 관리
print(ko.vocab().most_common(10) )
data = ko.vocab().most_common(500) # 자주 등장하는 단어 500개 선택
stopwords = ['점점', '지난', '측면', '기세', '북', '격진', '더비', '시오', '포체'
            '기점', '의', '이' , '당한', '로', '오른', '사이', '탈']
# 불용어 제거
tokens = [each_word for each_word in data if each_word not in stopwords]
tmp_data = dict(tokens)
# 상대적 10% 차이가 나도록 글씨의 사이즈를 조절
wordcloud = WordCloud(font_path=fontpath, relative_scaling = 0.1,
                     background_color='white',stopwords=stopwords,).generate_from_frequencies(tmp_data)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
# 의미가 중요하지 않고 카운트 하면 됨 - tfidf 바로 구할 수 있음

# konlpy, gensim embedding
from konlpy.tag import Okt
from gensim.models import Word2Vec, FastText
from collections import defaultdict
import numpy as np

docs = [
    "인공지능은 미래 산업의 핵심 기술이다.",
    "머신러닝은 인공지능을 구현하는 중요한 방법이다.",
    "딥러닝은 머신러닝의 한 분야이며 신경망을 사용한다.",
    "자연어처리는 언어 데이터를 분석하는 인공지능 기술이다.",
    "컴퓨터 비전은 이미지와 영상을 이해하는 인공지능 분야이다.",
    "추천 시스템은 사용자 취향을 분석하여 상품을 추천한다.",
    "데이터 분석은 패턴을 찾고 의사결정을 돕는다.",
    "딥러닝 모델은 대량의 데이터를 학습하여 예측한다.",
    "강화학습은 보상을 최대화하도록 행동을 학습한다.",
    "챗봇은 자연어처리와 인공지능 기술을 활용한다."
]

# 파싱하면서 인공지능 -> 인공, 지능
okt = Okt()
def tokenize(text):
  return [ # 품사, 어근 (먹다, 먹겠다, 먹는다 -> 먹다)
      word for word, pos in okt.pos(text, stem=True)
      if pos in ["Noun", "Verb", "Adjective"] and len(word) > 1
  ]
sentences = [tokenize(doc) for doc in docs]
print(sentences)

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100, # 표현 벡터 차원
    window=3, # 앞뒤 3개의 단어를 고려 학습
    min_count=1,
    sg=1, # 1: skip-gram(앞뒤 단어 간의 관계 속에서 예측), 0: BAG
    epochs=100, # deep learning (1세대 : 전체 데이터 학습)
    seed=42
)
# bigram을 지원하지 않고 철자 학습하지 않음
# print(w2v_model.wv.most_similar("인공지능", topn=5))

print(w2v_model.wv.most_similar("인공", topn=5))

ft_model = FastText( # 철자도 학습
    sentences=sentences,
    vector_size=100,
    window=3,
    min_count=1,
    sg=1, # skip-gram : 주변의 단어를 고려하면서
    epochs=100,
    seed=42
)
print(ft_model.wv.most_similar("인공지능", topn=5))

print(ft_model.wv.most_similar("인공지능기술", topn=5))

# deep learning
import torch
import torch.nn as nn
import torch.optim as optim

# 중복되지 않는 단어장
vocab = sorted(set(word for sent in sentences for word in sent))
# tokenToi
word2idx = {word: i for i, word in enumerate(vocab)}
# idTotoken
idx2word = {i: word for word, i in word2idx.items()}
vocab_size = len(vocab)
window_size = 3

# 동시 등장 확률
cooc = defaultdict(float) # dict + 숫자
for sent in sentences:
  # 문장을 숫자
  token_ids = [word2idx[w] for w in sent]
  # skip-gram에서 윈도사이즈 : 주변 단어를 고려
  # 센터에 선택
  # 중심위치, id 숫자(단어)
  # 0,1,2,3,4,,,,,,
  for center_pos, center_id in enumerate(token_ids):
    # 시작값은 max
    start = max(0, center_pos - window_size) # 예를 들어 첫 번째는 왼쪽이 없음
    # 종료값은 min
    end = min(len(token_ids), center_pos + window_size + 1)
    for context_pos in range(start, end):
      if center_pos == context_pos:
        continue
      context_id = token_ids[context_pos] # 해당 단어
      distance = abs(center_pos - context_pos) # 거리값
      cooc[(center_id, context_id)] += 1.0 / distance

# deep learning model
# 모델을 생성하려면 nn.Module을 상속
class SimpleGloVe(nn.Module):
  # vocab_size : 10
  # embedding_dim : 100
  def __init__(self, vocab_size, embedding_dim):
    super().__init__()
    # 가중치
    self.w = nn.Embedding(vocab_size, embedding_dim)
    # 가중치
    self.w_tilde = nn.Embedding(vocab_size, embedding_dim)
    # bias : deep learning에서는 나가는 차수 bias를 더해줌 (0으로 가지 않도록)
    self.b = nn.Embedding(vocab_size, 1)
    self.b_tilde = nn.Embedding(vocab_size, 1)
    # 행렬곱을 전진하면서 진행 (가장 기본망)
  def forward(self, i, j): # FFNN(feed forward: 먹이 주듯이 데이터 전달)
    wi = self.w(i)
    wj = self.w_tilde(j)
    bi = self.b(i).squeeze() # 차원축소
    bj = self.b_tilde(j).squeeze()
    return torch.sum(wi * wj, dim=1) + bi + bj

# 가중치 함수 (조절 함수)
# x값의 최고값을 100으로 제한
def weighting(x, xmax=100, alpha=0.75):
  return torch.where(x < xmax, (x / xmax) ** alpha, # 강도를 약하게
                     torch.ones_like(x)) # 1로 제한
embedding_dim = 100
glove = SimpleGloVe(vocab_size, embedding_dim)
# newton 최적화
# 경사하강법 -> Adam
# 최적화기능 : Adam의 기능 = learning-rate (처음은 크게, 나중은 작게)
                # momentum을 고려해서 - 가던 방향을 고려해서 -> zigzag로 가지말고 직진
# parameter는 DL 학습의 목적이 가중치 학습
optimizer = optim.Adam(glove.parameters(), lr=0.05)
# 동시 등장 확률
pairs = list(cooc.keys())
# 중심단어 벡터
i_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
# 문맥단어 벡터
j_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
# 가중한 값 (이웃과 얼마나 가까이 있는가)
x_ij = torch.tensor([cooc[p] for p in pairs], dtype=torch.float)

# learning-rate 조금씩 학습
for epoch in range(300): # epoch: 모든 데이터가 1번 학습하는 것
  optimizer.zero_grad() # 이전 epoch의 경사하강 학습을 초기화
  pred = glove(i_idx, j_idx) # 중심단어, 문맥단어인덱스
  # 예측값은 두 단어의 동시등장확률
  target = torch.log(x_ij) # 실제 동시 등장확률
  weight = weighting(x_ij) # 가중치 조절
  loss = torch.mean(weight * (pred - target) ** 2) # 손실함수
  loss.backward() # 역전파
  optimizer.step() # 학습 진행
  if (epoch + 1) % 50 == 0: # 50 epoch마다
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# glove의 목적
# 입력된 문장을 100차원의 벡터로 표현
# 10x100
def get_glove_vectors(model):
  with torch.no_grad(): # 학습하지 말고 
    vectors = model.w.weight + model.w_tilde.weight
    # DL에서는 tensor로 표현
  return vectors.numpy() # ndarray로 변환
glove_vectors = get_glove_vectors(glove)
# 10x100

def most_similar_glove(word, topn=5):
  if word not in word2idx:
    return f"{word}는 사전에 없습니다."
  idx = word2idx[word] # 단어 -> 인덱스 변환
  target_vec = glove_vectors[idx] # 100차원 1차원 벡터
  similarities = []
  for i, vec in enumerate(glove_vectors):
    if i == idx: # 자기자신이면 넘어가고
      continue
    # cosine 유사도
    sim = np.dot(target_vec, vec) / (
        np.linalg.norm(target_vec) * np.linalg.norm(vec)
    )
    similarities.append((idx2word[i], sim)) # 단어, 유사도
  # 유사도를 기준으로 역순으로 정렬
  return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]
print(most_similar_glove("인공", topn=5))

# 문제
# 웹페이지를 검색해서 word cloud를 작성하시오
# 메일에 붙여넣기로 제출하시오
import requests 
from bs4 import BeautifulSoup 
from konlpy.tag import Twitter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
res = requests.get('https://www.bbc.com/korean/articles/c1jy09x21w0o')
soup = BeautifulSoup(res.content, 'html.parser')
body = soup.select('.css-men093')[0]
text = " ".join(p.get_text() for p in body.find_all('p')) 
t = Twitter()
tokens_ko = t.nouns(text)
ko = nltk.Text(tokens_ko, name = '스포츠')
print(ko.vocab().most_common(30) )
data = ko.vocab().most_common(100)
stopwords = ['사람', '라며', '이번', '첫', '위해', '말', '앞', '씨', '생각', '것', '시간', '날', '시작', '명']
tokens = [(word, count) for word, count in data if word not in stopwords]
tmp_data = dict(tokens)
wordcloud = WordCloud(font_path=fontpath, relative_scaling = 0.1,
                     background_color='white',stopwords=stopwords,).generate_from_frequencies(tmp_data)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()