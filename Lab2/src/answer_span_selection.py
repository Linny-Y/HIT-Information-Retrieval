from sklearn.externals import joblib
from utils.metric import *
from ltp import LTP
import json
import os
from utils import evaluate
LTP_MODEL_PATH = '../data/data/base1.tgz'

TRAIN_DATA = '../data/data/train.json'

TEST_RESULT = '../data/output/test_answer_result.json'

TRAIN_ANSWER = '../data/output/train_answer.json'
TEST_ANSWER = '../data/output/test_answer.json'


ltp = LTP(LTP_MODEL_PATH)
def progress(is_train=False):

    clf = joblib.load(ROUGH_MODEL_PATH)
    tv = joblib.load(TF_IDF_MODEL_PATH)
    # 读取json文件
    path = TRAIN_DATA if is_train else TEST_RESULT
    with open(path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line.strip()) for line in f.readlines()]
    res = []
    none_cnt = 0
    for question in questions:
        if is_train:
            ans_sent = ' '.join(question['answer_sentence'])
        else:
            if len(question['answer_sentence']) == 0:
                question.pop('answer_pid')
                question.pop('answer_sentence')
                question['answer'] = ''
                res.append(question)
                continue
            ans_sent = question['answer_sentence'][0]
        sent = ' '.join(ltp.seg(question['question'])[0][0])
        test_data = tv.transform([sent])
        label = clf.predict(test_data)[0]
        ans_words = [word for word in ltp.seg(ans_sent)[0][0]]
        words_pos = ltp.pos(ans_words)
        if '：' in ans_sent or ':' in ans_sent:
            question['answer'] = ans_sent.split('：')[1] if '：' in ans_sent else ans_sent.split(':')[1]
        elif label == 'HUM':
            question['answer'] = pos_answer(ans_words, words_pos, ['nh', 'ni'])
        elif label == 'LOC':
            question['answer'] = pos_answer(ans_words, words_pos, ['nl', 'ns'])
        elif label == 'NUM':
            question['answer'] = pos_answer(ans_words, words_pos, ['m'])
        elif label == 'TIME':
            question['answer'] = pos_answer(ans_words, words_pos, ['nt'])
        else:
            question['answer'] = ''.join(ans_words)
        if not is_train:
            question.pop('answer_sentence')
        res.append(question)
        if question['answer'] == '':
            none_cnt += 1
    answer_path = TRAIN_ANSWER if is_train else TEST_ANSWER
    with open(answer_path, 'w', encoding='utf-8') as f:
        for sample in res:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def pos_answer(words, words_pos, pos):
    res = []
    for i in range(len(words_pos)):
        if words_pos[i] in pos:
            res.append(words[i])
    if len(res):
        return ''.join(res)
    else:
        return ''.join(words)


if __name__ == '__main__':
    progress(True)
    evaluate()
    progress()
