import numpy as np
import os
from preprocessed import  get_stop_words, remove_stop_words
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from ltp import LTP
LTP_MODEL_PATH = '../data/data/base1.tgz'

TRAIN_DATA_PATH = '../data/data/train_questions.txt'
TEST_DATA_PATH = '../data/data/test_questions.txt'

SEG_TRAIN_DATA_PATH = '../data/output/seg_train_questions.txt'
SEG_TEST_DATA_PATH = '../data/output/seg_test_questions.txt'

TEST_CLF_RESULT_PATH = '../data/output/clf_result_test_questions.txt'

stop_words = []

def get_data(rough, train_mode=True, remove_stopwords=False):
    ltp = LTP(LTP_MODEL_PATH)
    text, y = [], []
    if train_mode:
        path = TRAIN_DATA_PATH
        seg_path =  SEG_TRAIN_DATA_PATH
    else:
        path = TEST_DATA_PATH
        seg_path = SEG_TEST_DATA_PATH

    if os.path.exists(seg_path):
        for line in open(seg_path, 'r', encoding='utf-8'):
            parts = line.split("\t")
            if rough:
                y.append(parts[0].split('_')[0])
            else:
                y.append(parts[0])
            if remove_stopwords:
                pred_words = parts[1].split(' ')
                pred_words = remove_stop_words(pred_words)
                seg = ''
                for word in pred_words[:-1]:
                    seg += word + ' '
                seg += pred_words[-1]
                text.append(seg)
            else:
                text.append(parts[1])
        return text, y

    for line in open(path, 'r', encoding='utf-8'):
        parts = line.split('\t')
        # print(parts[1])
        if rough:
            tag = parts[0].split('_')[0]
        else:
            tag = parts[0]
        question = parts[1]

        pred_words = ltp.seg([question])[0][0]
        if remove_stopwords:
            pred_words = remove_stop_words(pred_words)
        seg = ''
        for word in pred_words[:-1]:
            seg += word + ' '
        seg += pred_words[-1]
        text.append(seg)
        y.append(tag)

    with open(seg_path, 'w', encoding='utf-8') as file:
        for tag, seg_question in zip(y, text):
            file.write(tag + '\t' + seg_question + '\n')
    return text, y


def train_predict_naive_bayes(rough = False):
    train_text, train_y = get_data(rough, True)
    test_text, test_y = get_data(rough, False)
    # 获得词频 再获得词频倒置文档频率
    count_vector = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_tfidf = TfidfTransformer().fit_transform(count_vector.fit_transform(train_text))
    # 分类器
    clf = MultinomialNB(alpha=0.001).fit(train_tfidf, train_y)
    predict = []
    for text in test_text:
        test_tfidf = TfidfTransformer().fit_transform(count_vector.transform([text]))
        predict.append(clf.predict(test_tfidf))
    # 计算准确率
    total, right = 0, 0
    for pre, real, text in zip(predict, test_y, test_text):
        total += 1
        if pre[0] == real: # pre = ['DES_OTHER']
            right += 1
        # else:
        #     print(str(pre[0]) + " " + str(real) + " " + str(text))
    return right * 1.0 / total

def train_predict_svm(rough = False):
    train_text, train_y = get_data(rough, True)
    test_text, test_y = get_data(rough, False)
    count_vector = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = count_vector.fit_transform(train_text)
    clf = svm.SVC(C=100.0, gamma=0.05)
    clf.fit(train_data, np.asarray(train_y))
    test_data = count_vector.transform(test_text)
    result = clf.predict(test_data)
    score = clf.score(test_data, test_y)
    if not rough:
        with open(TEST_CLF_RESULT_PATH, 'w', encoding='utf-8') as f:
            for r, y, q in zip(result, test_y, test_text):
                f.write(r + '\t' + y + '\t')
                list = q.split()
                for w in list:
                    f.write(w)
                f.write('\n')
    acc = {}
    total = {}
    for r, tag in zip(result, test_y):
        if tag not in acc.keys():
            acc[tag] = 0
            total[tag] = 0
        total[tag] += 1
        # print(r + ' ' + tag)
        if r == tag:
            acc[tag] += 1
    for tag in total.keys():
        # print(str(acc[tag]) + ' ' + str(total[tag]))
        acc[tag] = acc[tag] * 1.0 / total[tag]

    return acc, score

def print_score(acc, score):
    print("accuracy: {:.2%}".format(score))
    fmt = "{:20}\t{: >7.2%}"
    for tag in acc.keys():
        print(fmt.format(tag, acc[tag]))
    print()

def main():
    get_stop_words()
    # rate = train_predict_naive_bayes(False)
    # print("accuracy: {}".format(rate))
    # rate = train_predict_naive_bayes(True)
    # print("accuracy: {}".format(rate))
    acc, score = train_predict_svm()
    print("fine", end = ' ')
    print_score(acc, score)
    acc, score = train_predict_svm(True)
    print("rough", end = ' ')
    print_score(acc, score)


if __name__ == '__main__':
    main()