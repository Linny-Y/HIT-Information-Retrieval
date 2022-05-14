import os
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import urlretrieve
from copy import deepcopy


class Page(object):
    def __init__(self, url, title, paragraphs, file_name):
        """
        声明一个新的网页

        :param url: 网页url
        :param title: 网页标题, 字符串或分词列表
        :param paragraphs: 网页正文(取description), 字符串或分词列表
        :param file_name: 附件名称
        """
        self.data = {}
        self.data['url'] = url
        self.data['title'] = title
        self.data['paragraphs'] = paragraphs
        self.data['file_name'] = file_name

        self.data_seg = {}
        self.data_seg['url'] = url
        self.data_seg['segmented_title'] = ""
        self.data_seg['segmented_parapraghs'] = ""
        self.data_seg['file_name'] = file_name

    @staticmethod
    def load_url(url, file_path, attachment_type=('txt', 'doc', 'docx')):
        """
        从url中提取信息, 并保存到file_path中

        :param url: 提取url
        :param file_path: 存储附件的地址
        :param attachment_type: 下载附件类型
        :param tokenizer: 分词器, None不进行分词
        """
        root_url = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
                              url)[0]

        html = urlopen(url).read().decode('utf-8')
        soup = BeautifulSoup(html, features='html.parser')

        # 提取标题和正文
        title = soup.title.string.strip()
        file_path = file_path + '/' + title
        description = soup.find(attrs={"name": "description"})
        if description is None:
            paragraphs = title
        else:
            paragraphs = description['content']

        # 提取并下载附件
        all_href = soup.find_all('a')
        file_name = []
        download_url = []
        for h in all_href:
            name = h.get_text()
            if name.endswith(attachment_type):
                current_url = h['href']
                download_url.append(root_url + current_url)
                file_name.append(name)

        if len(file_name):
            if not (os.path.exists(file_path)):
                os.makedirs(file_path)

            for i in range(len(file_name)):
                file_path_now = file_path + '/' + file_name[i]
                urlretrieve(download_url[i], file_path_now)  # 下载文件
        else:
            file_path = None
        return Page(url, title, paragraphs, file_name)

    def tokenize(self, tokenizer):
        """
        对该网页的title和paragraphs进行分词处理
        分词结果覆盖原数据

        :param tokenizer: 分词器
        """
        self.data_seg['segmented_title'] = tokenizer(self.data['title'])
        self.data_seg['segmented_paragraphs'] = tokenizer(self.data['paragraphs'])

