import json
import time
import numpy as np
from scipy.linalg import norm
from ltp import LTP
from gensim.summarization import bm25
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer

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


def build_feature(train_mode=True):
    """
    从初始数据中抽取特征
    :param train_mode: 训练模式标记
    :return: 将提取到的特征写入文件
    """
    # 读取train/查询结果文件
    path = TRAIN_DATA if train_mode else SEARCH_RESULT
    with open(path, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f.readlines()]
    if not train_mode:
        questions.sort(key=lambda q: q['qid'])  # 按qid升序排序

    # 读入分词后文件
    passage = {}
    with open(SEG_DATA_PATH, encoding='utf-8') as f:
        for line in f.readlines():
            read = json.loads(line)
            passage[read['pid']] = read['document']
    # 读入原文件
    passage_raw = {}
    with open(DATA_PATH, encoding='utf-8') as f:
        for line in f.readlines():
            read = json.loads(line)
            passage_raw[read['pid']] = read['document']

    # 建立特征矩阵
    feature = []
    result = []
    for k in range(len(questions)):
        question = questions[k]
        sentences, corpus = [], []
        # 获取每个问题对应的文件的分词后的文档内容
        # 计算词频 词频倒置文档频率
        # 获取所有句子corpus以训练bm25模型
        if train_mode:
            cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
            for sen in passage[question['pid']]:
                sentences.append(' '.join(sen))
            cv.fit(sentences)
            tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tv.fit(sentences)
            corpus.extend(passage[question['pid']])
        else:
            if len(question['answer_pid']) == 0:  # 没有检索到文档
                print("no answer pid: {}".format(question['qid']))
                continue
            for pid in question['answer_pid']:
                for sen in passage[pid]:
                    sentences.append(' '.join(sen))
                corpus.extend(passage[pid])
            cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
            cv.fit(sentences)
            tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tv.fit(sentences)

        # 提取 BM25 特征
        bm25_model = bm25.BM25(corpus)
        # 对当前问题进行分词
        q = ltp.seg([question['question']])[0][0]
        # 计算当前问题与对应文档中的句子的相关度
        scores = bm25_model.get_scores(q)
        
        # 获取特征 获取答案句子
        if train_mode:
            for i in range(len(passage[question['pid']])):
                ans_sen = passage[question['pid']][i]
                feature_array = extract_feature(q, ans_sen, cv, tv)
                feature_array.append(scores[i])
                feature.append(' '.join([str(fea) for fea in feature_array]) + '\n')
                sen = {}
                if passage_raw[question['pid']][i] in question['answer_sentence']:
                    sen['label'] = 1
                else:
                    sen['label'] = 0
                sen['qid'] = question['qid']
                sen['question'] = question['question']
                sen['answer'] = passage[question['pid']][i]
                result.append(sen)
            # print(feature)
            # assert 1==0
        else:
            for i in range(len(sentences)):
                feature_array = extract_feature(q, sentences[i], cv, tv)
                feature_array.append(scores[i])
                feature.append(' '.join([str(fea) for fea in feature_array]) + '\n')
                sen = {'label': 0, 'qid': question['qid'], 'question': question['question'], 'answer': sentences[i]}
                result.append(sen)
    # 特征写入文件
    feature_path = TRAIN_FEATURE if train_mode else TEST_FEATURE
    with open(feature_path, 'w', encoding='utf-8') as f:
        f.writelines(feature)
    # 句子写入文件
    sentence_path = TRAIN_SENTENCE if train_mode else TEST_SENTENCE
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for sample in result:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def extract_feature(question, answer, cv, tv):
    """
    抽取句子的特征
    答案句特征: 答案句长度; 是否含冒号
    答案句和问句之间的特征: 问句和答案句词数差异; uni-gram词共现比例; 词频cv向量相似度; tf-idf向量相似度; bm25相似度
    :param question: 问题
    :param answer: 答案
    :param cv: Count Vector
    :param tv: Tf-idf Vector
    :return: 特征列表
    """
    feature = []
    len_answer, len_question = len(answer), len(question)
    feature.append(len_answer)
    feature.append(1) if '：' in answer else feature.append(0)
    feature.append(abs(len_question - len_answer))
    feature.append(len(set(question) & set(answer)) / float(len(set(question))))
    vectors = cv.transform([' '.join(question), ' '.join(answer)]).toarray()
    cosine_similar = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]) + 1)
    feature.append(cosine_similar if not np.isnan(cosine_similar) else 0.0)
    vectors = tv.transform([' '.join(question), ' '.join(answer)]).toarray()
    tf_sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]) + 1)
    feature.append(tf_sim if not np.isnan(tf_sim) else 0)
    return feature

def generate_svm_rank_data(train_mode=True):
    """
    为SVM Rank生成训练和测试数据
    :param train_mode: 训练模式
    :return:
    """
    if train_mode:
        sentence_path, feature_path = TRAIN_SENTENCE, TRAIN_FEATURE
    else:
        sentence_path, feature_path = TEST_SENTENCE, TEST_FEATURE
    sentences = []
    for line in open(sentence_path, 'r', encoding='utf-8'):
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
        index = [0, 1, 2, 3, 4, 5, 6]
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

def calculate_mrr():
    """
    计算开发集上的完美匹配率 mrr
    :return:
    """
    with open(SVM_RANK_TRAIN_RESULT, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    dev = []
    with open(SVM_RANK_DEV_DATA, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            # i label pid
            dev.append((i, int(line[0]), int(line.split(' ')[1].split(':')[1])))
            i += 1
    q_s = 0
    question_num = 0
    question_with_answer = 0
    prefect_correct = 0
    mrr = 0.0
    old_qid = dev[0][2]
    for i in range(len(dev) + 1):
        if i == len(dev) or dev[i][2] != old_qid:  # 前i-1-q_s个为相同的qid
            p = np.argsort(-predictions[q_s:i]) + q_s # 获取排序后的下标
            for k in range(len(p)):
                if dev[p[k]][1] == 1: # 答案相符
                    question_with_answer += 1
                    if k == 0:        # 且排序为第一个
                        prefect_correct += 1
                    mrr += 1.0 / float(k + 1)
                    break
            if not i == len(dev):
                q_s = i
                old_qid = dev[i][2]
            question_num += 1
    print("question num:{}, question with answer:{}, prefect_correct:{}, MRR:{}"
          .format(question_num, question_with_answer, prefect_correct, mrr / question_num))

def get_test_ans():
    """
    将SVM rank的结果转化为特征答案句
    :return:
    """
    with open(SVM_RANK_TEST_RESULT, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    
    with open(TEST_SENTENCE, 'r', encoding='utf-8') as f:
        items = [json.loads(line.strip()) for line in f.readlines()]
    
    sen_qid = []
    for item in items:
        sen_qid.append(item['qid'])

    with open(SEARCH_RESULT, 'r', encoding='utf-8') as f:
        test_res = [json.loads(line.strip()) for line in f.readlines()]

    for res in test_res:
        if res['qid'] not in sen_qid:
            res['answer_sentence'] = []
            continue
        s_begin = sen_qid.index(res['qid']) # 获取当前问题的答案句子第一次出现的位置
        s_end = s_begin
        while s_end < len(sen_qid) and sen_qid[s_end] == res['qid']:
            s_end += 1 
        p = np.argsort(-predictions[s_begin:s_end])
        answer = []
        for i in p[0:3]: # 取前三个
            answer.append(items[i + s_begin]['answer'])
        res['answer_sentence'] = answer
    with open(TEST_RESULT, 'w', encoding='utf-8') as f:
        for sample in test_res:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    # # train
    # build_feature()
    # # test
    # build_feature(False)
    
    # train
    # generate_svm_rank_data()
    # # test
    # generate_svm_rank_data(False)

    """
        在命令行中使用svm_rank_windows进行训练
        训练使用的命令: 
        ./svm_rank_learn -c 10.0  ../output/svm_train.txt model_10.dat

        使用训练好的模型预测dev集: 
        ./svm_rank_classify ../output/svm_dev.txt model_10.dat ../output/dev_predictions

        使用训练好的模型预测test集:
        ./svm_rank_classify ../output/svm_test.txt model_10.dat ../output/test_predictions
    """

    calculate_mrr()
    # get_test_ans()