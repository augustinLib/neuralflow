import numpy as np
import copy
from collections import OrderedDict

def tokenization(text: str) -> tuple:
    """
    sentence tokenization based on space

    Parameters
    ----------
    text (str) : sentence to be tokenized

    """
    text = text.lower()
    punctuation_list = [".", ",", "?", "!"]
    for punc in punctuation_list:
        text = text.replace(f"{punc}", f" {punc}")
    words = text.split(" ")

    word_to_id = OrderedDict()
    id_to_word = OrderedDict()

    for i, word in enumerate(set(words)):
        word_to_id[word], id_to_word[i] = i, word

    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word


class BaseTokenizer():
    def __init__(self):
        self.word_to_id = OrderedDict()
        self.id_to_word = OrderedDict()
        self.punctuation_list = [".", ",", "?", "!"]
        self.special_token = OrderedDict()
        self.special_token["unk"] = "[UNK]"
        token_value = list(self.special_token.values())
        for i, token in enumerate(token_value):
            self.word_to_id[token], self.id_to_word[i] = i, token

        self.n_tokens = len(self.special_token.keys())

    
    def add_speical_token(self, name, token):
        self.special_token[name] = token

        


class SpaceTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()


    def train(self, text: list):
        """
        sentence tokenization based on space

        Parameters
        ----------
        text () : sentence to be tokenized

        """
        if isinstance(text, str):
            text = text.lower()
            for punc in self.punctuation_list:
                text = text.replace(f"{punc}", f" {punc}")
            words = text.split(" ")

        else:
            text_copy = copy.deepcopy(text)
            words = []
            for index, sentence in enumerate(text_copy):
                text_copy[index] = sentence.lower()
                for punc in self.punctuation_list:
                    text_copy[index] = text_copy[index].replace(f"{punc}", f" {punc}")
                words += text_copy[index].split(" ")

        for i, word in enumerate(set(words)):
                self.word_to_id[word], self.id_to_word[i+self.n_tokens] = i+self.n_tokens, word

            
    def encode(self, text: str) -> list:
        for punc in self.punctuation_list:
            text = text.replace(f"{punc}", f" {punc}")
        words = text.split(" ")
        encoded_text = np.array([])

        for word in words:
            try:
                encoded_text = np.append(encoded_text, word)
            except IndexError:
                encoded_text = np.append(encoded_text, self.special_token["unk"])

        return encoded_text

    
    def decode(self, id_list: list) -> str:
        id_list_copy = copy.deepcopy(id_list)
        for index, id in enumerate(id_list_copy):
            id_list_copy[index] = self.id_to_word[id]
        
        decoded_id = " ".join(id_list_copy)
        return decoded_id

    
    def get_vocab(self):
        return self.word_to_id


    def get_id(self):
        return self.id_to_word

