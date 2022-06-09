import os
import math
import json
import numpy as np
from win32com import client as wc

from ltp import LTP

LTP_MODEL_PATH = '../data/base1.tgz'
DATA_DIR = '../data/craw/craw.json'
PRE_PROCESSED_DATA_DIR = '../data/seg_web.json'
STOP_WORDS = '../data/stopwords(new).txt'
FILE_DIR = 'E:\\Courses\\8_2022_spring\\Information Retrieval\\Lab\\Lab3\\data\\craw\\attachment'
BM25_PAGE_DIR = '../data/bm25_page.json'
BM25_FILE_DIR = '../data/bm25_file.json'
ltp = LTP(LTP_MODEL_PATH)


class BM25:
    def __init__(self, docs, path=None):
        if path is not None and os.path.exists(path):
            self.load(path)
            return
        # list of document length in docs
        self.doc_len_list = [len(doc) for doc in docs]
        # average document length in docs
        self.avg_len_list = sum(self.doc_len_list) / len(self.doc_len_list)
        # frequency of words
        self.tf = []
        # IDF
        self.idf = {}
        # DF
        self.df = {}
        for doc in docs: 
            freq = {}
            for word in doc: # 建立词频
                freq[word] = 1 if word not in freq else freq[word] + 1
            self.tf.append(freq)

            for word, f in freq.items(): # 建立文件频率 包含某词的文件数目
                self.df[word] = 1 if word not in self.df else self.df[word] + 1

        for word, f in self.df.items(): # 计算idf
            self.idf[word] = math.log(len(self.doc_len_list) - math.log(f))

    def save(self, path):
        bm25 = {'doc_len_list': self.doc_len_list, 'avg_len_list': self.avg_len_list,
                'doc_word_freq': self.tf, 'idf': self.idf}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(bm25, f)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            bm25 = json.load(f)
        self.doc_len_list = bm25['doc_len_list']
        self.avg_len_list = bm25['avg_len_list']
        self.tf = bm25['doc_word_freq']
        self.idf = bm25['idf']

    def score(self, query, specific_range=None):
        '''
            相关性计算
        '''
        scores = []
        k = 2
        b = 0.5
        if specific_range is None:
            for idx in range(len(self.doc_len_list)):
                tf = self.tf[idx]
                score = sum([self.idf[word] * tf[word] * (k + 1) /
                             (tf[word] + k * (1 - b + b * self.doc_len_list[idx] / self.avg_len_list))
                             for word in query if word in tf])
                scores.append(score)
        else:
            for idx in range(len(self.doc_len_list)):
                tf = self.tf[idx]
                score = sum([self.idf[word] * tf[word] * (k + 1) /
                             (tf[word] + k * (1 - b + b * self.doc_len_list[idx] / self.avg_len_list))
                             for word in query if word in tf]) if idx in specific_range else 0
                scores.append(score)
        return scores


class InvertedIndex:
    def __init__(self, docs):
        '''
            建立索引
        '''
        self.dic = {}
        for idx, sentence in enumerate(docs):
            for word in sentence:
                if word not in self.dic:
                    self.dic[word] = set()
                self.dic[word].add(idx)

    def search(self, query):
        '''
            在词典中查找索引
        '''
        ret = set()
        for word in query:
            if word in self.dic:
                ret = ret | self.dic[word]
        return ret



def load(): 
    '''
        加载分词后的爬取的文件
    '''
    with open(PRE_PROCESSED_DATA_DIR, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]


class Search:
    def __init__(self):
        self.pages = load()
        self.files, docs_page, docs_file = [], [], []
        for index, page in enumerate(self.pages):
            self.files += [(index, i) for i in range(len(page['file_name']))] # 页面编号 文件编号
            docs_page.append(page['segmented_title'] + page['segmented_paragraphs'])
            for file, file_content in zip(page['segmented_file_name'], page['segmented_file_contents']):
                docs_file.append(page['segmented_title'] + file + file_content) 

        self.bm25_page = BM25(docs_page, BM25_PAGE_DIR)
        self.bm25_page.save(BM25_PAGE_DIR)
        self.index_page = InvertedIndex(docs_page)

        self.bm25_file = BM25(docs_file, BM25_FILE_DIR)
        self.bm25_file.save(BM25_FILE_DIR)
        self.index_file = InvertedIndex(docs_file)

    def search(self, query, authority, mode='page'):
        query = ltp.seg([query])[0][0]
        if mode == 'page':
            spec = self.index_page.search(query)
            scores = self.bm25_page.score(query, spec)
        else:
            spec = self.index_file.search(query)
            scores = self.bm25_file.score(query, spec)
        sorted_scores = np.argsort(-np.array(scores)).tolist()
        # idx = sorted_scores.index(0)
        idx = sorted(scores, reverse = True).index(0)
        sorted_scores = sorted_scores[:idx]
        if mode == 'page':
            return [self.pages[index] for index in sorted_scores if self.pages[index]['authority'] <= authority]
        else:
            ret = [self.files[idx] for idx in sorted_scores]
            return [(self.pages[page_idx]['file_name'][file_idx], self.pages[page_idx]['title'],
                     self.pages[page_idx]['authority']) for page_idx, file_idx in ret
                    if self.pages[page_idx]['authority'] <= authority]


def convert_doc2docx():
    # 将doc文件转换成docx保存
    folders = os.listdir(FILE_DIR)
    word = wc.Dispatch('Word.Application')
    for folder in folders:
        files = os.listdir(FILE_DIR + '\\' + folder)
        for f in files:
            if 'docx' in f:
                continue
            if 'doc' in f:
                doc = word.Documents.Open(FILE_DIR + '\\' + folder + '\\' + f)
                if not os.path.exists(FILE_DIR + '\\' + folder + '\\' + f + 'x'):
                    doc.SaveAs(FILE_DIR + '\\' + folder + '\\' + f + 'x', 12, False, "", True, "", False, False, False, False)
                doc.Close()
    word.Quit()


if __name__ == '__main__':
    convert_doc2docx()
