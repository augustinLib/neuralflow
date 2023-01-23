import numpy as np
import copy
from collections import OrderedDict


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

    
    def __repr__(self) -> str:
        return "Tokenizer"
        

class SpaceTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()


    def train(self, text):
        """
        train vocab based on space

        Parameters
        ----------
        text (str or iterable) : sentence to be tokenized

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
                sentence = sentence.lower()
                for punc in self.punctuation_list:
                    sentence = sentence.replace(f"{punc}", f" {punc}")
                words += sentence.split(" ")

        for i, word in enumerate(set(words)):
                self.word_to_id[word], self.id_to_word[i+self.n_tokens] = i+self.n_tokens, word

            
    def encode(self, text):
        """
        convert text into integer id
        """
        encoded_text = np.array([]).astype(np.int32)
        
        if isinstance(text, str):
            text = text.lower()
            for punc in self.punctuation_list:
                text = text.replace(f"{punc}", f" {punc}")
            words = text.split(" ")
        
            for word in words:
                try:
                    encoded_word = self.word_to_id[word]
                    encoded_text = np.append(encoded_text, encoded_word)
                except KeyError:
                    encoded_text = np.append(encoded_text, self.word_to_id["[UNK]"])
                    

        else:
            text_copy = copy.deepcopy(text)

            for index, sentence in enumerate(text_copy):
                sentence = sentence.lower()
                for punc in self.punctuation_list:
                    sentence = sentence.replace(f"{punc}", f" {punc}")
                    sentence.split(" ")

                encoded_sentence = np.array([])
                for word in sentence:
                    try:
                        encoded_word = self.word_to_id[word]
                        encoded_sentence = np.append(encoded_sentence, encoded_word)
                    except KeyError:
                        encoded_sentence = np.append(encoded_sentence, self.word_to_id["[UNK]"])
                
                encoded_text = np.append(encoded_text, encoded_sentence)
        
        return encoded_text

    
    def decode(self, id_list: list) -> str:
        id_list_copy= copy.deepcopy(id_list)
        decoded_id = []
        if id_list_copy.ndim == 1:
            for index, id in enumerate(id_list_copy):
                temp = self.id_to_word[id]
                decoded_id.append(temp)

            id_sentence = " ".join(decoded_id)

            return id_sentence

        else:
            temp_decoded = []
            for id_sentence in id_list_copy:
                for index, id in enumerate(id_sentence):
                    temp_decoded.append(self.id_to_word[id])

                temp_decoded_str = " ".join(temp_decoded)
                decoded_id.append(temp_decoded_str)

            return decoded_id

    
    def get_vocab(self):
        return self.word_to_id


    def get_id(self):
        return self.id_to_word


def cosine_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)

    return np.dot(nx, ny)