import json
import os
import time
import numpy as np
from scipy.linalg import norm
from multiprocessing import Pool
from ltp import LTP
from gensim.summarization import bm25
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib

from preprocessed import  get_stop_words, remove_stop_words
LTP_MODEL_PATH = '../data/data/base1.tgz'

DATA_PATH = '../data/data/passages_multi_sentences.json'
TRAIN_DATA = '../data/data/train.json'
TEST_DATA = '../data/data/test.json'

SEG_DATA_PATH = '../data/output/seg_passages_multi_sentences.json'
INDEX_PATH = '../data/output/index.txt'

SEARCH_RESULT = '../data/output/test_search_result.json'
TRAIN_FEATURE = '../data/output/feature.txt'
TRAIN_SENTENCE = '../data/output/all_sentence.txt'
TRAIN_BM25 = '../data/output/train_bm25.txt'
TEST_FEATURE = '../data/output/test_feature.txt'
TEST_SENTENCE = '../data/output/test_sentence.txt'
SVM_RANK_TRAIN_DATA = '../data/output/svm_train.txt'
SVM_RANK_DEV_DATA = '../data/output/svm_dev.txt'
SVM_RANK_TEST_DATA = '../data/output/svm_test.txt'
RANK_RESULT = '../data/output/svm_result.txt'
SVM_RANK_TRAIN_RESULT = '../data/output/dev_predictions'
SVM_RANK_TEST_RESULT = '../data/output/test_predictions'
TEST_RESULT = '../data/output/test_answer_result.json'


ltp = LTP(LTP_MODEL_PATH)

def make_seg_data():
    with open(SEG_DATA_PATH, 'w', encoding='utf-8') as f:
        for line in open(DATA_PATH, 'r', encoding='utf-8'):
            passage = json.loads(line)
            pid = passage['pid']
            if pid % 1000 == 0:
                print(pid)
            passage['document'] = ltp.seg(passage['document'])[0]
            f.write(json.dumps(passage, ensure_ascii=False) + '\n')

def BM25_search(train_mode=False):
    get_stop_words()
    passages = []
    for line in open(SEG_DATA_PATH, 'r', encoding='utf-8'):
        passages.append(json.loads(line))
    # corpus = [' '.join(passage['document']).split(' ') for passage in passages]
    corpus = []
    for passage in passages:
        corpus.append(sen for sen in passage['document'])

    bm25_model = bm25.BM25(corpus)

    passages_raw = []
    path = TRAIN_DATA if train_mode else TEST_DATA
    for line in open(path, 'r', encoding='utf-8'):
        passages_raw.append(json.loads(line))

    # test for bm25 search
    if train_mode:
        pid_true, pid_predict = [], []
        for passage in passages_raw:
            question = remove_stop_words(ltp.seg(passage['question'])[0][0])
            scores = bm25_model.get_scores(question)
            sorted_scores = np.argsort(-np.array(scores))
            if sum(np.array(scores) != 0) > 0:
                pid_predict.append([sorted_scores[0]])
            else:
                pid_predict.append([])
        # evaluate
        # match, num = 0, len(pid_true)
        # for i in range(num):
        #     if pid_true[i] in pid_predict[i]:
        #         match += 1
        # acc = match * 1.0 / num
        # # 0.8707025411061285
        # print('acc: ' + str(acc))
    else:
        for passage in passages_raw:
            question = remove_stop_words(ltp.seg(passage['question'])[0][0])
            scores = bm25_model.get_scores(question)
            sorted_scores = np.argsort(-np.array(scores))
            if sum(np.array(scores) != 0) > 0:
                passage['answer_pid'] = [int(idx) for idx in sorted_scores[0:3]]
            else:
                passage['answer_pid'] = []
        with open(SEARCH_RESULT, 'w', encoding='utf-8') as f:
            for passage in passages_raw:
                f.write(json.dumps(passage, ensure_ascii=False) + '\n')

def build_feature(train_mode=True):
    """
    从初始数据中抽取特征
    :param train_mode: 训练模式标记
    :return: 将提取到的特征写入文件
    """
    # 读取train json文件
    if train_mode:
        with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
            questions = [json.loads(line.strip()) for line in f.readlines()]
    else:
        with open(SEARCH_RESULT, 'r', encoding='utf-8') as f:
            questions = [json.loads(line.strip()) for line in f.readlines()]
        questions.sort(key=lambda item_: item_['qid'])  # 按qid升序排序
    # 读入passage json文件
    passage = {}
    with open(SEG_DATA_PATH, encoding='utf-8') as f:
        for line in f.readlines():
            read = json.loads(line.strip())
            passage[read['pid']] = read['document']
    # 读入raw passage json文件
    passage_raw = {}
    with open(DATA_PATH, encoding='utf-8') as f:
        for line in f.readlines():
            read = json.loads(line.strip())
            passage_raw[read['pid']] = read['document']

    # 建立特征矩阵
    feature = []
    ret = []

    for k in range(len(questions)):
        question = questions[k]
        sents, corpus = [], []
        if train_mode:
            cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
            cv.fit(passage[question['pid']])
            tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tv.fit(passage[question['pid']])
            for sent in passage[question['pid']]:
                corpus.append(sent.split())

        else:
            for pid in question['answer_pid']:
                sents += passage[pid]
                for sent in passage[pid]:
                    corpus.append(sent.split())
            if len(sents) == 0:  # 没有检索到文档
                print("no answer pid: {}".format(question['qid']))
                continue
            cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
            cv.fit(sents)
            tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tv.fit(sents)

        # 提取 BM25 特征
        bm25_model = bm25.BM25(corpus)
        q = ltp.seg(question['question'])[0][0]
        scores = bm25_model.get_scores(q)

        if train_mode:
            for i in range(len(passage[question['pid']])):
                ans_sent = passage[question['pid']][i]
                feature_array = extract_feature(q, ans_sent, cv, tv)
                feature_array.append(scores[i])
                feature.append(' '.join([str(attr) for attr in feature_array]) + '\n')
                sen = {}
                if passage_raw[question['pid']][i] in question['answer_sentence']:
                    sen['label'] = 1
                else:
                    sen['label'] = 0
                sen['qid'] = question['qid']
                sen['question'] = question['question']
                sen['answer'] = passage[question['pid']][i]
                ret.append(sen)
        else:
            for i in range(len(sents)):
                feature_array = extract_feature(q, sents[i], cv, tv)
                feature_array.append(scores[i])
                feature.append(' '.join([str(attr) for attr in feature_array]) + '\n')
                sen = {'label': 0, 'qid': question['qid'], 'question': question['question'], 'answer': sents[i]}
                ret.append(sen)
    # 特征写入文件
    feature_path = TRAIN_FEATURE if train_mode else TEST_FEATURE
    with open(feature_path, 'w', encoding='utf-8') as f:
        f.writelines(feature)
    # 句子写入文件
    sentence_path = TRAIN_SENTENCE if train_mode else TEST_SENTENCE
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for sample in ret:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    ltp.release()


def generate_svm_rank_data(train_mode=True):
    """
    为SVM Rank生成训练和测试数据
    :param train_mode: 训练模式
    :return:
    """
    if train_mode:
        sen_path, feature_path = TRAIN_SENTENCE, TRAIN_FEATURE
    else:
        sen_path, feature_path = TEST_SENTENCE, TEST_FEATURE
    sentences = []
    for line in open(sen_path, 'r', encoding='utf-8'):
        sentences.append(json.loads(line.strip()))
    # 读取特征文件
    features = []
    for line in open(feature_path, 'r', encoding='utf-8'):
        features.append(line.strip().split(' '))
    assert len(sentences) == len(features), 'Something wrong!'
    data, data_qid, qid_set = [], [], set()
    train_index = int(0.8 * float(5352))
    flag = False
    for k in range(len(sentences)):
        item = sentences[k]
        qid_set.add(item['qid'])
        # 写入训练集文件
        if len(qid_set) >= train_index + 1 and train_mode and flag is False:
            sort_index = np.argsort(data_qid)
            with open(SVM_RANK_TRAIN_DATA, 'w', encoding='utf-8') as f:
                for j in range(len(sort_index)):
                    f.write(data[sort_index[j]])
            flag = True
            data.clear()
            data_qid.clear()
        feature_array = features[k]
        index = [0, 1, 2, 3, 4, 5, 6, 7]
        feature = ["{}:{}".format(j + 1, feature_array[index[j]]) for j in range(len(index))]
        data.append("{} qid:{} {}\n".format(item['label'], item['qid'], ' '.join(feature)))
        data_qid.append(item['qid'])
    # 写入开发集
    if train_mode:
        sort_index = np.argsort(data_qid)
        with open(SVM_RANK_DEV_DATA, 'w', encoding='utf-8') as f:
            for j in range(len(sort_index)):
                f.write(data[sort_index[j]])
    else:
        with open(SVM_RANK_TEST_DATA, 'w', encoding='utf-8') as f:
            f.writelines(data)


def extract_feature(question, answer, cv, tv):
    """
    抽取句子的特征
    答案句特征：答案句长度；是否含冒号
    答案句和问句之间的特征：问句和答案句词数差异；uni-gram词共现比例；字符共现比例；
                        词频cv向量相似度；tf-idf向量相似度；bm25相似度
    :param question: 问题
    :param answer: 答案
    :param cv: Count Vector
    :param tv: Tf-idf Vector
    :return: 特征列表
    """
    feature = []
    answer_words = answer.split(' ')
    len_answer, len_question = len(answer_words), len(question)
    feature.append(len_answer)
    feature.append(1) if '：' in answer else feature.append(0)
    feature.append(abs(len_question - len_answer))
    feature.append(len(set(question) & set(answer_words)) / float(len(set(question))))
    feature.append(len(set(question) & set(''.join(answer_words))) / float(len(set(question))))
    vectors = cv.transform([' '.join(question), answer]).toarray()
    cosine_similar = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(cosine_similar if not np.isnan(cosine_similar) else 0.0)
    vectors = tv.transform([' '.join(question), answer]).toarray()
    tf_sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(tf_sim if not np.isnan(tf_sim) else 0)
    return feature


def get_test_ans():
    """
    将SVM rank的结果转化为特征答案句
    :return:
    """
    with open(SVM_RANK_TEST_RESULT, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    print(len(predictions))
    with open(TEST_SENTENCE, 'r', encoding='utf-8') as f:
        items = [json.loads(line.strip()) for line in f.readlines()]
    print(len(items))
    sent_qid = []
    for item in items:
        sent_qid.append(item['qid'])
    with open(SEARCH_RESULT, 'r', encoding='utf-8') as f:
        test_res = [json.loads(line.strip()) for line in f.readlines()]
    for res in test_res:
        if res['qid'] not in sent_qid:
            res['answer_sentence'] = []
            continue
        s = sent_qid.index(res['qid'])
        e = s
        while e < len(sent_qid) and sent_qid[e] == res['qid']:
            e += 1
        p = np.argsort(-predictions[s:e])
        answer = []
        for i in p[0:3]:
            answer.append(items[i + s]['answer'])
        res['answer_sentence'] = answer
    with open(TEST_RESULT, 'w', encoding='utf-8') as f:
        for sample in test_res:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # make_seg_data()
    BM25_search(True)
    # BM25_search()