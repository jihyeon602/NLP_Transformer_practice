# 경고문을 무시합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
from konlpy.tag import Kkma, Okt  #한국어 형태소 사전 기반 한궉어 단어 추출 라이브러리

# sts-train.tsv 파일에 저장되어 있는 KorSTS 데이터셋을 불러옵니다.
sent = pd.read_table("NLP_practice4/q1/sts-train.tsv", delimiter='\t', header=0)['sentence1']

# sent 변수에 저장된 첫 5개 문장을 확인해봅니다.
print(sent[:5]) #시리즈

# 꼬꼬마 형태소 사전을 이용해서 sent 내 문장의 명사를 nouns 리스트에 저장하세요.
kkma = Kkma()
nouns = []
for s in sent:
    try:
        nouns.extend(kkma.nouns(s)) #리스트 적재
    except Exception as e:
        print("오류 발생")

# 명사의 종류를 확인해봅니다.
print('✅명사종류확인: ',set(nouns))

# Open Korean Text 형태소 사전을 이용해서 sent 내 형태소 분석 결과를 pos_results 리스트에 저장하세요.
okt = Okt()
pos_results = []
# 문장별로 형태소 분석 (POS 태깅)
for s in sent:
    try:
        pos = okt.pos(s)  # ('형태소', '품사') 튜플 리스트
        pos_results.append(pos)
    except Exception as e:
        print(f"오류 발생")

# 분석 결과를 확인해봅니다.
print('✅형태소분석: ', pos_results)

# stemming 기반 형태소 분석이 적용된 sent의 두 번째 문장을 stem_pos_results 리스트에 저장하세요.
stem_pos_results = []
stem_pos_results.append(okt.pos(sent[1]))
print('')
print('✅형태소분석_2: ', stem_pos_results)