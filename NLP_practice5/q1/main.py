import re
from sklearn.feature_extraction.text import CountVectorizer

regex = re.compile('[^a-z ]')

with open("NLP_practice5/q1/text.txt", 'r', encoding='utf-8') as f:
    documents = []
    for line in f:
        # doucments 리스트에 리뷰 데이터를 저장하세요.
        line = line.rstrip('\n').lower()
        clean_line = regex.sub('', line)
        if clean_line.strip():
            documents.append(clean_line)
        
print(documents[:2])

# CountVectorizer() 객체를 이용해 Bag of words 문서 벡터를 생성하여 변수 X에 저장하세요.  
countVC = CountVectorizer()
X = countVC.fit_transform(documents)

# 변수 X의 차원을 변수 dim에 저장하세요.
dim = X.shape
# X 변수의 차원을 확인해봅니다.
print("X 차원:", dim) #456, 12136

# 위에서 생성한 CountVectorizer() 객체에서 첫 10개의 칼럼이 의미하는 단어를 words_feature 변수에 저장하세요.
words_feature = countVC.get_feature_names_out()[:10]
# CountVectorizer() 객체의 첫 10개 칼럼이 의미하는 단어를 확인해봅니다.
print("첫 10개 피처 단어:",words_feature)

# 단어 "comedy"를 의미하는 칼럼의 인덱스 값을 idx 변수에 저장하세요.
idx = countVC.vocabulary_['comedy']
# 단어 "comedy"의 인덱스를 확인합니다.
print('comedy 인덱스', idx)

# 첫 번째 문서의 Bag of words 벡터를 vec1 변수에 저장하세요.
vec1 = X[0 , :]
# 첫 번째 문서의 Bag of words 벡터를 확인합니다.
print(vec1)