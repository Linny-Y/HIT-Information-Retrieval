import joblib
from utils.metric import *
from ltp import LTP
import json
import os
LTP_MODEL_PATH = '../data/data/base1.tgz'

TRAIN_DATA = '../data/data/train.json'

TEST_RESULT = '../data/output/test_answer_result.json'

TRAIN_ANSWER = '../data/output/train_answer.json'
TEST_ANSWER = '../data/output/test_answer.json'

MODEL_ROUGH_PATH = '../data/output/model_rough'
MODEL_TF_IDF_PATH = '../data/output/model_tf_idf'

ltp = LTP(LTP_MODEL_PATH)
def progress(train_mode=False):

    clf = joblib.load(MODEL_ROUGH_PATH)
    tv = joblib.load(MODEL_TF_IDF_PATH)
    # 读取json文件
    path = TRAIN_DATA if train_mode else TEST_RESULT
    with open(path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line.strip()) for line in f.readlines()]

    res = []
    none_cnt = 0
    for question in questions:
        if train_mode:
            ans_sent = ' '.join(question['answer_sentence'])
            print(question['answer_sentence'])
            print(ans_sent)
            assert 1==0
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
        if not train_mode:
            question.pop('answer_sentence')
        res.append(question)
        if question['answer'] == '':
            none_cnt += 1
    answer_path = TRAIN_ANSWER if train_mode else TEST_ANSWER
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

def evaluate():
    """
    使用老师给出的metrics来计算bleu得分
    :return: 评测结果
    """
    with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line.strip()) for line in f.readlines()]
    with open(TRAIN_ANSWER, 'r', encoding='utf-8') as f:
        train_ans = [json.loads(line.strip()) for line in f.readlines()]
    cnt = len(train_data)
    all_prediction = []
    all_ground_truth = []
    bleu = 0.0
    p = 0.0
    r = 0.0
    f1 = 0.0
    for i in range(cnt):
        bleu += bleu1(train_ans[i]['answer'], train_data[i]['answer'])
        p_1, r_1, f1_1 = precision_recall_f1(train_ans[i]['answer'], train_data[i]['answer'])
        p += p_1
        r += r_1
        f1 += f1_1
        all_prediction.append(train_ans[i]['answer'])
        all_ground_truth.append(train_data[i]['answer'])
    em = exact_match(all_prediction, all_ground_truth)
    print("bleu1:{}, exact_match:{},\np:{}, r:{}, f1:{}".format(bleu / cnt, em, p / cnt, r / cnt, f1 / cnt))

if __name__ == '__main__':
    progress(True)
    evaluate()
    progress()
