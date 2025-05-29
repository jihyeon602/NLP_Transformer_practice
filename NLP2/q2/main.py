import re

word_counter = dict()
regex = re.compile('[^a-zA-Z]')

# 텍스트 파일을 소문자로 변환 및 숫자 및 특수기호를 제거한 딕셔너리를 만드세요.
with open('250527/q2/text.txt', 'r', encoding= 'utf-8') as f: # 실습 1 과 동일한 방식으로 `IMDB dataset`을 불러옵니다.
    for line in f:
        for word in line.rstrip('\n').split():
            word = regex.sub('', word.lower()) #특수문자 제거, 소문자로 변경
            if word:
                word_counter[word] = word_counter.get(word, 0) + 1 #word 키가 word_counter 딕셔너리에 존재하면 해당 값 반환, 아니면 0


#print(word_counter)
# 단어 "the"의 빈도수를 확인해 보세요.
count = word_counter.get('the', 0)

print(count)