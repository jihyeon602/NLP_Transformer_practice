# 가상환경 사용: python=3.8 // numpy==1.23.5 // tensorflow==2.10.0 // scikit-learn==1.2.2

# 경고문을 무시합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pickle
import tensorflow as tf
import numpy as np


# 학습된 모델을 불러오는 함수입니다.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,), batch_size=batch_size),  # 명시적 Input 레이어 추가
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform'
        ),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# 학습된 모델에서 문장을 생성하는 함수입니다.
def generate_text(model, start_string):
    num_generate = 100

    # 예측할 문자 혹은 문자열의 정수형 인덱스로 변환
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        # 다음 발생확률이 제일 높은 문자로 예측
        predicted_id = np.argmax(predictions[-1])
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# 기존 학습한 모델의 구조를 불러옵니다.
# 예측을 위해 batch_size는 1로 조절되었습니다.
model = build_model(vocab_size=65, embedding_dim=256, rnn_units=1024, batch_size=1)

# model.load_weights()을 이용해 데이터를 불러오세요.
model.load_weights(tf.train.latest_checkpoint('NLP_practice5/q6/q6_checkpoints')).expect_partial()
model.build([1, None])

# char2idx, idx2char는 주어진 문자를 정수 인덱스로 매핑하는 딕셔너리 입니다.
with open('NLP_practice5/q6/word_index.pkl', 'rb') as f:
    char2idx, idx2char = pickle.load(f)

# "Juliet: "이라는 문자열을 추가하여 생성된 문장을 result 변수에 저장하세요.
result = generate_text(model=model, start_string= "Juliet: ")
print(result)
'''
Juliet: the oracle
That we remember me and my servant speak.

KING RICHARD III:
Some day to hear the truth o
'''