# -*- coding: utf-8 -*-
import random
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import sqrt, dot

random.seed(10)

doc1 = ["homelessness has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school work or vote for the matter"]

doc2 = ["it may have ends that do not tie together particularly well but it is still a compelling enough story to stick with"]

# 데이터를 불러오는 함수입니다.
def load_data(filepath):
    regex = re.compile('[^a-z ]')

    gensim_input = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            lowered_sent = line.rstrip().lower()
            filtered_sent = regex.sub('', lowered_sent)
            tagged_doc = TaggedDocument(filtered_sent, [idx])
            gensim_input.append(tagged_doc)
            
    return gensim_input
    
def cal_cosine_sim(v1, v2):
    # 벡터 간 코사인 유사도를 계산해 주는 함수를 완성합니다.
    dot_product = dot(v1, v2)

    # 벡터 크기 (norm)
    norm_v1 = sqrt(dot(v1, v1))
    norm_v2 = sqrt(dot(v2, v2))

    # 코사인 유사도 계산
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return cosine_similarity
    
# doc2vec 모델을 documents 리스트를 이용해 학습하세요.
documents = load_data("NLP_practice5/q4/text.txt")
model = Doc2Vec(window=2, vector_size= 50, epochs= 5)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

# 학습된 모델을 이용해 doc1과 doc2에 들어있는 문서의 임베딩 벡터를 생성하여 각각 변수 vector1과 vector2에 저장하세요.
# 토큰화 필수
vector1 = model.infer_vector(doc1[0].lower().split())
vector2 = model.infer_vector(doc1[0].lower().split())

# vector1과 vector2의 코사인 유사도를 변수 sim에 저장하세요.
sim = cal_cosine_sim(vector1, vector2)
# 계산한 코사인 유사도를 확인합니다.
print(f"코사인 유사도: {sim:.4f}") #0.3782