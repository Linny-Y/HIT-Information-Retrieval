from utils.tokenizer import Tokenizer
from utils.base import Base

if __name__ == '__main__':
    base = Base.load_json('../data/results/craw.json')
    tokenizer = Tokenizer('../data/stopwords/stopwords(new).txt')

    base.tokenize(tokenizer)
    base.save_json('../data/results/segment.json')
    base.save_json('../data/results/preprocessed.json')
