import json
import os
import time
import numpy as np
from multiprocessing import Pool
from ltp import LTP
LTP_MODEL_PATH = '../data/data/base1.tgz'

STOP_WORDS_PATH = '../data/data/stopwords(new).txt'
DATA_PATH = '../data/data/passages_multi_sentences.json'

INDEX_PATH = '../data/output/index.txt'

stop_words = []
word_dict = {}
ltp = LTP(LTP_MODEL_PATH)

def get_stop_words():
    """
    从指定的文件中获取stopwords
    :return: 文件不存在则报错, 存在则返回stopwords列表
    """
    path = STOP_WORDS_PATH
    if not os.path.exists(path):
        print("No stop words file!")
        return
    for line in open(path, "r", encoding="utf-8"):
        stop_words.append(line.strip())
    # print(len(stop_words))

def remove_stop_words(text_words: list):
    """
    对分词结果进行去停用词处理
    :param text_words: 分词列表
    :return: 去掉停用词后的分词结果
    """
    ret = []
    for text_word in text_words:
        if text_word not in stop_words:
            ret.append(text_word)
    # print(ret)
    return ret

def preprocess():
    # 读取索引
    if os.path.exists(INDEX_PATH):
        print("Loading index\n")
        for line in open(INDEX_PATH, 'r', encoding='utf-8'):
            li = line.split(": ")
            word = li[0]
            pid_list = li[1].strip()[:-1].split(",")
            word_dict[word] = set(pid_list)
        return
    start = time.time()
    for line in open(DATA_PATH, 'r', encoding='utf-8'):
        passage = json.loads(line)
        pid = passage['pid']
        if pid % 1000 == 0: # 进度
            print(pid, end=" ")
            print(time.time() - start)

        pred_words = [remove_stop_words(x) for x in ltp.seg(passage['document'])[0]]
        # print(pred_words)
        # assert 1==0
        for ws in pred_words:
            for w in ws:
                if w not in word_dict.keys():  #建立倒排索引
                    word_dict[w] = set()
                word_dict[w].add(pid) # 添加索引
        
    with open(INDEX_PATH, 'w', encoding='utf-8') as index_file:
        for word, indexes in word_dict.items():
            index_file.write(str(word) + ': ')
            
            for i in indexes:
                index_file.write(str(i) + ',')
            index_file.write('\n')
    

def search():
    """
    A naive search system.
    :return:
    """
    print("*************************************************")
    print("* 检索系统说明:                                 *")
    print("* 使用以下符号连接检索词:                       *")
    print("*     &&:与                                     *")
    print("*     ||:或                                     *")
    print("* p.s.只支持全&&或全||, 不支持混合式            *")
    print("* 未使用&&或||时, 默认进行分词并默认与模式查询  *")
    print("* 退出: exit                                    *")
    print("*************************************************")
    while True:
        print("输入检索词: ", end="")
        question = input()
        if question == 'exit':
            print("Exit!")
            break
        and_mode = True
        if '&&' in question:
            word_list = question.split('&&')
        elif '||' in question:
            word_list = question.split('||')
            and_mode = False
        else:
            word_list = remove_stop_words(ltp.seg([question])[0][0])
        count = []
        for word in word_list:
            if word in word_dict:
                count.append(word_dict[word])

        length = len(count)
        result = count[0]
        if and_mode: # 交
            if length > 1:
                for i in range(1, length):
                    result = result.intersection(count[i])
        else: # 并
            if length > 1:
                for i in range(1, length):
                    result = result.union(count[i])
        result = list(result)
        print("Find pid: ", end="")
        if len(result):
            for res in result[:-1]:
                print(res + ", ", end='')
            print(result[-1])    
        else:
            print("Did not find.")

def main():
    get_stop_words()
    preprocess()
    search()

if __name__ == '__main__':
    main()