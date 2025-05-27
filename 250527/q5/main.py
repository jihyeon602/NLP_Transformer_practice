from gensim.models import FastText
import pandas as pd

# Emotions dataset for NLP 데이터셋을 불러오는 load_data() 함수입니다.
def load_data(filepath):
    data = pd.read_csv(filepath, delimiter=';', header=None, names=['sentence','emotion'])
    data = data['sentence']

    gensim_input = []
    for text in data:
        gensim_input.append(text.rstrip().split())

    return gensim_input

input_data = load_data("250527/q5/emotions_train.txt")

# fastText 모델을 학습하세요.
# input_data에 저장되어 있는 텍스트 데이터를 사용해서 단어별 문맥의 길이를 의미하는 window는 3, 벡터의 차원이 100, 단어의 최소 발생 빈도를 의미하는 min_count가 10인 fastText 모델을 학습하세요.
# epochs는 10으로 설정합니다.

ft_model = FastText(min_count=10, window=3, vector_size=100)
ft_model.build_vocab(input_data)
ft_model.train(input_data, total_examples=ft_model.corpus_count, epochs=10)


# day와 유사한 단어 10개를 확인하세요.
similar_day = ft_model.wv.most_similar("day")

print('🌅 ',similar_day)

# night와 유사한 단어 10개를 확인하세요.
similar_night = ft_model.wv.most_similar("night")

print('🌃 ',similar_night)

# elllllllice의 임베딩 벡터를 확인하세요.
wv_elice = ft_model.wv['elllllllice']

print(wv_elice)