data = ['this is a dog', 'this is a cat', 'this is my horse','my name is elice', 'my name is hank']

def count_unigram(docs):
    unigram_counter = dict()
    # docs에서 발생하는 모든 unigram의 빈도수를 딕셔너리 unigram_counter에 저장하여 반환하세요.
    for sentence in docs:
        for word in sentence.split():
            unigram_counter[word] = unigram_counter.get(word, 0) + 1
        return unigram_counter
    
    return unigram_counter

def count_bigram(docs):
    bigram_counter = dict()
    # docs에서 발생하는 모든 bigram의 빈도수를 딕셔너리 bigram_counter에 저장하여 반환하세요.
    for sentence in docs:
        words = sentence.split()
        for w1, w2 in zip(words[:-1], words[1:]):
            bigram = (w1, w2)
            bigram_counter[bigram] = bigram_counter.get(bigram, 0) + 1

    return bigram_counter

# Laplace Smoothing (라플라스 스무딩) 적용
# : 등장하지 않은 n-gram에도 작은 확률을 부여, 전체 문장 확률이 무조건 0이 되는 문제 방지
def cal_prob(sent, unigram_counter, bigram_counter, vocab_size=0, smoothing=True):
    words = sent.split()
    result = 1.0
    # sent의 발생 확률을 계산하여 변수 result에 저장 후 반환하세요.
    if vocab_size == 0:
        vocab_size = len(unigram_counter)

    for w1, w2 in zip(words[:-1], words[1:]):
        bigram = (w1, w2)
        bigram_count = bigram_counter.get(bigram, 0)
        unigram_count = unigram_counter.get(w1, 0)

        if smoothing:
            prob = (bigram_count + 1) / (unigram_count + vocab_size)
        else:
            if unigram_count == 0 or bigram_count == 0:
                return 0.0
            prob = bigram_count / unigram_count

        result *= prob

    return result

# 주어진data를 이용해 unigram 빈도수, bigram 빈도수를 구하고 "this is elice" 문장의 발생 확률을 계산해봅니다.
unigram_counter = count_unigram(data)
bigram_counter = count_bigram(data)
print(cal_prob("my name is elice", unigram_counter, bigram_counter))
print(cal_prob("my name is dog", unigram_counter, bigram_counter))
print(cal_prob("this is elice", unigram_counter, bigram_counter)) 
