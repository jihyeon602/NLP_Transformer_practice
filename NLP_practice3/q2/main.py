import pandas as pd

def cal_partial_freq(texts, emotion):
    partial_freq = dict()
    filtered_texts = texts[texts['emotion']==emotion]
    filtered_texts = filtered_texts['sentence']
    
    # 전체 데이터 내 각 단어별 빈도수를 입력해 주는 부분을 구현하세요.
    for sentence in filtered_texts:
        for word in sentence.split():
            partial_freq[word] = partial_freq.get(word, 0) + 1

    return partial_freq

def cal_total_freq(partial_freq):
    # partial_freq 딕셔너리에서 감정별로 문서 내 전체 단어의 빈도수를 계산하여 반환하는 부분을 구현하세요.
    total = sum(partial_freq.values())
    return total

# Emotions dataset for NLP를 불러옵니다.
data = pd.read_csv("NLP_practice3/q2/emotions_train.txt", delimiter=';', header=None, names=['sentence','emotion'])

# happy가 joy라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
partial_freq_joy = cal_partial_freq(data, 'joy')
happy_count_in_joy = partial_freq_joy.get('happy', 0)
joy_total = cal_total_freq(partial_freq_joy)
joy_likelihood = happy_count_in_joy / joy_total if joy_total > 0 else 0
print(round(joy_likelihood, 4))

# happy가 sadness라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
partial_freq_sad = cal_partial_freq(data, 'sad')
happy_count_in_sad = partial_freq_sad.get('happy', 0)
sad_total = cal_total_freq(partial_freq_sad)
sad_likelihood = happy_count_in_sad / sad_total if sad_total > 0 else 0
print(round(sad_likelihood, 4))

# can이 surprise라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
partial_freq_surp = cal_partial_freq(data, 'surprise')
can_count_in_surp = partial_freq_surp.get('can', 0)
surp_total = cal_total_freq(partial_freq_surp)
sup_likelihood = can_count_in_surp / surp_total if surp_total > 0 else 0
print(round(sup_likelihood, 4))
