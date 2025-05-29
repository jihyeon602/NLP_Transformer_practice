from sklearn.model_selection import train_test_split

# 파일을 읽어오세요.
data = []

with open('NLP_practice3/q1/emotions_train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        # 줄 끝 개행 제거 후 ; 기준 분리
        line = line.rstrip('\n')
        if ';' in line:
            sentence, emotion = line.split(';', 1)
            data.append((sentence.strip(), emotion.strip()))


# 읽어온 파일을 학습 데이터와 평가 데이터로 분할하세요.
train, test = train_test_split(data, test_size=0.2, random_state=7)


# 학습 데이터셋의 문장과 감정을 분리하세요.
Xtrain = [sentence for sentence, emotion in train]
Ytrain = [emotion for sentence, emotion in train]

print(Xtrain[:5], len(Xtrain))
print(set(Ytrain)) #중복제거

# 평가 데이터셋의 문장과 감정을 분리하세요.
Xtest = [sentence for sentence, emotion in test]
Ytest = [emotion for sentence, emotion in test]

print(Xtest[:5], len(Xtest))
print(set(Ytest)) #중복제거