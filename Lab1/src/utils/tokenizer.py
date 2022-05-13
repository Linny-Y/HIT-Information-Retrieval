from ltp import LTP
from typing import List


class Tokenizer(object):
    def __init__(self, stopwords_path):
        self.__tokenizer = LTP()

        self.__stopwords = []  # 导入停用词表
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.__stopwords.append(line.strip())

    def __call__(self, paragraphs):
        """对给定的文段进行分词处理

        Args:
            paragraphs: 待分词文段

        Returns:
            分词文段
        """
        if isinstance(paragraphs, str):  # str
            paragraphs = paragraphs.strip()
            if len(paragraphs) == 0:
                return []
            # print(paragraphs)
            return self.__tokenize_list([paragraphs])[0]
        else: 
            return self.__tokenize_list(paragraphs)

    def __tokenize_list(self, paragraph):
        ret, _ = self.__tokenizer.seg(paragraph)
        ans = []
        ret = ret[0]
        # for p in ret:
        #     current = []
        #     for w in p:
        #         if not(w in self.__stopwords):
        #             current.append(w)
        #     ans.append(current)
        current = []
        for w in ret:
            if not(w in self.__stopwords):
                current.append(w)
        ans.append(current)
        return ans
