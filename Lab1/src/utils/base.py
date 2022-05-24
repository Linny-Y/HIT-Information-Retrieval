from utils.page import Page
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import json
import os
import json
from bs4 import BeautifulSoup
from urllib.request import urlopen


class Base(object):
    def __init__(self, pages):
        self.pages = pages

    @staticmethod
    def load_url(root_url, file_path, num_max, ends=('htm')):
        """
        以root_url为根路径进行BFS, 提取最少num_max个网页的信息
        选择根目录中以.htm作为结尾的子网页遍历

        :param root_url: 检索根路径
        :param file_path: 网页附件保存地址
        :param num_max: 检索网页最少数量
        :param ends: 检索网页的结尾特征
        """

        pool = Pool()  # 进程池
        rec = {}
        current = [root_url]
        pages = []
        num = 0

        rec[root_url] = True
        while num <= num_max:
            if len(current) == 0:
                break
            num += len(current)
            # print(num, len(pages))

            jobs = []
            for u in current:   # 获取页面内容
                jobs.append(pool.apply_async(Page.load_url, args=(
                    u, file_path, )))
            for i in jobs:
                try:
                    pages.append(i.get())
                except Exception:
                    continue
            # for u in current:
            #     pages.append(Page.load_url(u, file_path))
            print(len(pages), end=" ")

            htmls = []
            jobs = []
            jobs = [pool.apply_async(
                crawl, args=(u, ends, )) for u in current]  # 获取页面中超链接
            for i in jobs:
                htmls.extend(i.get())
            # for u in current:
            #     htmls.extend(crawl(u, ends))
            # print(htmls)
            current = []
            for html in htmls:
                if 'http' in html:
                    continue
                html = root_url + html

                if html not in rec:
                    rec[html] = True
                    current.append(html)
        pool.close()
        pool.join()
        # print(len(pages))
        return Base(pages)

    def tokenize(self, tokenizer):
        """利用分词器对文本进行分词

        Args:
            tokenizer: 分词器
        """
        print("tokenizing")
        for i in tqdm(range(len(self.pages))):
            (self.pages[i]).tokenize(tokenizer)

    def save_json(self, output_path):
        """
        将网页库导出为json文件的形式

        :param output_path: 输出路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, page in enumerate(self.pages):
                if 'seg' in output_path:
                    json.dump(page.data_seg, f, ensure_ascii=False)
                elif 'pre' in output_path:   # 输出前10行
                    if i < 10:
                        json.dump(page.data_seg, f, ensure_ascii=False)
                    else:
                        break
                else:
                    json.dump(page.data, f, ensure_ascii=False)
                f.write('\n')

    @staticmethod
    def load_json(input_path):
        """从json文件中导入网页库

        Args:
            input_path: 导入路径
        """
        pages = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                current = json.loads(line)
                pages.append(Page(current['url'], current['title'],
                                  current['paragraphs'], current['file_name']))
        return Base(pages)


def crawl(url, ends):
    html = urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html, features='html.parser')

    ret = []
    all_href = soup.find_all('a')  # 超链接标签
    for h in all_href:
        try:
            next_url = h['href']  # href属性
            # print(next_url)
        except:
            continue

        if next_url.endswith(ends):
            ret.append(next_url)

    return ret


if __name__ == '__main__':
    test_url = 'http://jwc.hit.edu.cn/'
    save_path = 'Lab1/data/attachment'
    output_path = 'Lab1/data/result/craw.json'

    base = Base.load_url(test_url, save_path, 10)
    base.save_json(output_path)
