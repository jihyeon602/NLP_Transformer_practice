import pandas as pd
from gensim.models import Word2Vec

def load_data(filepath):
    data = pd.read_csv(filepath, delimiter=';', header=None, names=['sentence','emotion'])
    data = data['sentence']

    gensim_input = []
    for text in data:
        gensim_input.append(text.rstrip().split())
    return gensim_input

input_data = load_data("250527/q5/emotions_train.txt")

# word2vec 모델을 학습하세요.
#input_data에 저장되어 있는 텍스트 데이터를 사용해서 단어별 문맥의 길이를 의미하는 window는 2, 벡터의 차원이 300인 word2vec 모델을 학습하세요. (epochs는 10으로 설정)
w2v_model = Word2Vec(min_count=1, window=2, vector_size= 300)
w2v_model.build_vocab(input_data)
w2v_model.train(input_data, total_examples= w2v_model.corpus_count, epochs=10)

# happy와 유사한 단어를 확인하세요.
similar_happy = w2v_model.wv.most_similar("happy")

print('😁 ',similar_happy)

# sad와 유사한 단어를 확인하세요.
similar_sad = w2v_model.wv.most_similar("sad")
print('😂 ',similar_sad)

# 단어 good과 bad의 임베딩 벡터 간 유사도를 확인하세요.
similar_good_bad = w2v_model.wv.similarity("good", "bad")

print('⬆️⬇️ ',similar_good_bad)

# 단어 sad과 lonely의 임베딩 벡터 간 유사도를 확인하세요.
similar_sad_lonely = w2v_model.wv.similarity("sad", "lonely")

print('😂🥲 ',similar_sad_lonely)

# happy의 임베딩 벡터를 확인하세요.
wv_happy = w2v_model.wv["happy"]

print('😁 ',wv_happy)
