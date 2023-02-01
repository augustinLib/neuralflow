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


def cos_similarity(x, y, eps=1e-8):
    '''코사인 유사도 산출

    :param x: 벡터
    :param y: 벡터
    :param eps: '0으로 나누기'를 방지하기 위한 작은 값
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s(을)를 찾을 수 없습니다.' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''유사 단어 검색

    :param query: 쿼리(텍스트)
    :param word_to_id: 단어에서 단어 ID로 변환하는 딕셔너리
    :param id_to_word: 단어 ID에서 단어로 변환하는 딕셔너리
    :param word_matrix: 단어 벡터를 정리한 행렬. 각 행에 해당 단어 벡터가 저장되어 있다고 가정한다.
    :param top: 상위 몇 개까지 출력할 지 지정
    '''
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환

    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot
            
            

def create_contexts_target(corpus, window_size=1):
    '''맥락과 타깃 생성

    :param corpus: 말뭉치(단어 ID 목록)
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)