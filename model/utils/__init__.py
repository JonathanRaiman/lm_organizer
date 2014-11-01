from collections import Counter
import gzip, pickle, itertools
from epub_conversion.utils import get_files_from_path
from xml_cleaner import to_raw_text_markupless
from .documenttree import DocumentTree

def import_mini_wiki_corpus(path):
    filepaths = get_files_from_path("txt", path)
    documents = {}
    for path, name in filepaths:
        # magic numbers 15 and 324 remove copyright and repeated info at end
        documents[name] = open(path, "rt").read()[15:-324]
    return documents

def code_to_index(max_code_len, code):
    offset = 0
    for b in code:
        if b == 1:
            offset += (2 ** max_code_len)
        else:
            offset += 1
        max_code_len -= 1
    return offset
    
def collect_counts(documents):
    vocab = Counter()
    for value in documents.values():
        vocab.update(word for sentence in to_raw_text_markupless(value) for word in sentence)
    return vocab

def save_object(obj, path):
    with gzip.open(path, "w") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(path):
    f =  gzip.open(path, "r")
    obj = pickle.load(f)
    f.close()
    return obj

def codes_for_depth(depth):
    return [code for k in range(depth+1) for code in generate_codes(depth, k) ]


def codes_upto_depth(depth):
    return [code for n in range(depth+1) for code in codes_for_depth(n)]

def generate_codes(n, trues):
    result = []
    for bits in itertools.combinations(range(n), trues):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        result.append(s)
    return result


def number_of_branches_per_depth(depth):
    """

    Sum of powers of 2 is:

    2^0 + 2^1 + 2^2 +  ... + 2^k = 2^k+1 - 1

    """
    return 2**(depth+1) - 1


__all__ = [
    "DocumentTree",
    "load_vocab",
    "save_vocab",
    "collect_counts",
    "import_mini_wiki_corpus",
    "number_of_branches_per_depth",
    "code_to_index",
    "codes_for_depth",
    "codes_upto_depth",
    "save_object",
    "load_object"
    ]