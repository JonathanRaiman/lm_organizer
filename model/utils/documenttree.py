import numpy as np
from xml_cleaner import to_raw_text_markupless

class DocumentTree:
    def __init__(self, name, document, vocab, index):
        self.index = index
        self.name  = name
        self.words = np.array([vocab.get(word) for sentence in to_raw_text_markupless(document) for word in sentence if vocab.get(word)], dtype='int32')
        self.size = len(self.words)
        
    def create_example_window(self, code):
        window_size = self.size
        offset = 0
        for b in code:
            window_size /= 2
            offset += b * window_size
        return (int(offset), int(window_size / 2))
    
    def create_example(self, code, top = True, sample_size = 3):
        """
        Create an example for a document by using a binary
        code for accessing the document slicing hierarchy.

        [] => full document
        [0] => document first half
        [1] => document second half
        ...
        [0,1] => second-half of document first half
        etc.

        Inputs
        ------

        code list<int>  : the binary code for accessing a slice
        sample_size int : how many words to get from that document piece.
        top boolean     : sample from first half or second half of slice.

        Outputs
        -------

        nparray <int32> : the indices of the words sampled.

        """
        offset, size = self.create_example_window(code)
        word_slice = self.words[offset:offset+size] if top else self.words[offset+size:offset+2*size]
        return np.random.choice(word_slice, sample_size)
