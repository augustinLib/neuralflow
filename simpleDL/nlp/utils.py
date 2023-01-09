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
        
        """
        encoded_text = np.array([])
        
        if isinstance(text, str):
            text = text.lower()
            for punc in self.punctuation_list:
                text = text.replace(f"{punc}", f" {punc}")
            words = text.split(" ")
        
            for word in words:
                try:
                    encoded_text = np.append(encoded_text, word)
                except IndexError:
                    encoded_text = np.append(encoded_text, self.speical_token["unk"])
                    

        else:
            text_copy = copy.deepcopy(text)

            for index, sentence in enumerate(text_copy):
                sentence = sentence.lower()
                for punc in self.punctuation_list:
                    sentence = sentence.replace(f"{punc}", f" {punc}")
                    sentence.split(" ")

                for word in sentence:
                    try:
                        
                        encoded_text[index] = np.append(encoded_text, word)
                    except IndexError:
                        text_copy[index] = text_copy[index].replace(f"{punc}", f" {punc}")



        
        
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

