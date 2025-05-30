from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2

sent = '트와이스 아이오아이 좋아여 tt가 저번에 1위 했었죠?'

# 학습에 사용할 데이터가 train_data에 저장되어 있습니다.
corpus_path = 'NLP_practice4/q2/q2_articles/articles.txt'
train_data = DoublespaceLineCorpus(corpus_path)
print("학습 문서의 개수: %d" %(len(train_data))) #30091

# LRNounExtractor_v2 객체를 이용해 train_data에서 명사로 추정되는 단어를 nouns 변수에 저장하세요.
noun_extractor = LRNounExtractor_v2(verbose=True, min_num_of_features = 2, max_right_length=5, extract_compound = True, ensure_normalized=True)

# 명사 추출 학습 수행
nouns_score = noun_extractor.train_extract(train_data)  # train_data는 str 리스트여야 함

# 추출된 명사 저장
nouns = list(nouns_score.keys())

# 확인
print(f"상위 10개 명사 출력: {nouns[:10]}")  # 상위 10개 명사 출력

# 생성된 명사의 개수를 확인해봅니다.
print(f"생성 명사 개수: {len(nouns)}")

# 생성된 명사 목록을 사용해서 sent에 주어진 문장에서 명사를 sent_nouns 리스트에 저장하세요.
sent_nouns = []
for s in sent:
    extracted = [word for word in nouns if word in s]
    sent_nouns.append(extracted)

print(f"주어진 문장 명사 확인: {sent_nouns}")