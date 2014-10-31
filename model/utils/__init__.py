from collections import Counter
import gzip, pickle, numpy as np
from epub_conversion.utils import get_files_from_path
from xml_cleaner import to_raw_text

def import_mini_wiki_corpus(path):
    filepaths = get_files_from_path("txt", path)
    documents = {}
    for path, name in filepaths:
        # magic numbers 15 and 324 remove copyright and repeated info at end
        documents[name] = open(path, "rt").read()[15:-324]
    return documents
    
def collect_counts(documents):
    vocab = Counter()
    for value in documents.values():
        vocab.update(word for sentence in to_raw_text(value) for word in sentence)
    return vocab

def save_vocab(vocab, path):
    with gzip.open(path, "w") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

def load_vocab(path):
    f =  gzip.open(path, "r")
    vocab = pickle.load(f)
    f.close()
    return vocab
        
class DocumentTree:
    def __init__(self, document, vocab, index):
        self.index = index
        self.words = np.array([vocab.get(word) for sentence in to_raw_text(document) for word in sentence if vocab.get(word)], dtype='int32')
        self.size = len(self.words)
        
    def create_example_window(self, code):
        window_size = self.size
        offset = 0
        for b in code:
            window_size //= 2
            offset += b * window_size
        return (offset, window_size)
    
    def create_example(self, code, sample_size = 3):
        offset, size = self.create_example_window(code)
        return np.random.choice(self.words[offset:offset+size], sample_size)