# 경고문을 무시합니다.
import warnings
warnings.filterwarnings(action='ignore')

from flask import Flask, request, Response
import pickle
import json
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query = json_['infer_texts']
    
    # 학습된 객체 cv 변수와 clf 변수를 이용해 전달된 문자열의 감정을 예측하는 코드를 작성하세요.
    # 전달된 문장을 CountVectorizer로 변환
    vec = cv.transform(query)
    
    # 예측 수행
    preds = clf.predict(vec)    

    # 예측된 결과를 response 딕셔너리에 "문서의 순서: 예측된 감점" 형태로 저장하세요.
    response = {f"{i+1}번 문장": pred for i, pred in enumerate(preds)}


    return Response(json.dumps(response, ensure_ascii=False), content_type="application/json")

if __name__ == '__main__':
    with open('NLP_practice3/q5/nb_model.pkl', 'rb') as f:
        cv, clf = pickle.load(f)
    
    app.run(host='127.0.0.1', port=8080)