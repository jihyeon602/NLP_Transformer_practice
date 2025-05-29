import pandas as pd
import numpy as np

def cal_partial_freq(texts, emotion):
    filtered_texts = texts[texts['emotion'] == emotion]
    filtered_texts = filtered_texts['sentence']
    partial_freq = dict()

    # 실습 2에서 구현한 부분을 완성하세요.
    for sentence in filtered_texts:
        for word in sentence.split():
            partial_freq[word] = partial_freq.get(word, 0) + 1

    return partial_freq

def cal_total_freq(partial_freq):
    total = sum(partial_freq.values())
    return total

def cal_prior_prob(data, emotion):
    filtered_texts = data[data['emotion'] == emotion]
    # data 내 특정 감정의 로그발생 확률을 반환하는 부분을 구현하세요.
    emotion_count = len(filtered_texts)
    total_count = len(data)
    emotion_log = np.log(emotion_count / total_count)
    return emotion_log

def predict_emotion(sent, data):
    emotions = ['anger', 'love', 'sadness', 'fear', 'joy', 'surprise']
    predictions = []
    smoothing = 10
    data = 'NLP_practice3/q3/emotions_train.txt'
    train_txt = pd.read_csv(data, delimiter=';', header=None, names=['sentence', 'emotion'])

    # sent의 각 감정별 로그 확률을 predictions 리스트에 저장하세요.
    for emotion in emotions:
        # 빈도 계산
        partial_freq = cal_partial_freq(train_txt, emotion)
        total = cal_total_freq(partial_freq)
        prior_log = cal_prior_prob(train_txt, emotion)

        log_prob = prior_log  # 초기값: 사전 확률

        for word in sent.split():
            word_freq = partial_freq.get(word, 0)
            # 로그 가능도 계산 (스무딩 포함)
            word_log_prob = np.log((word_freq + smoothing) / (total + smoothing * (len(partial_freq) + 1)))
            log_prob += word_log_prob

        predictions.append((emotion, log_prob))
    # 감정별 로그 확률 출력 (선택사항)
    for emotion, prob in predictions:
        print(f"{emotion}: {round(prob, 4)}")

    return max(predictions, key=lambda x: x[1])

# 아래 문장의 예측된 감정을 확인하세요.
test_sent = "i really want to go and enjoy this party"
predicted = predict_emotion(test_sent, "NLP_practice3/q2/emotions_train.txt")
print('가장높게 예측된 감정:', predicted) #결과: ('joy', -55.9027180740051)
