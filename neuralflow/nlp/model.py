from copy import deepcopy
from collections import OrderedDict
from neuralflow.gpu import *
from neuralflow.function import *
from neuralflow.model import *


class LanguageModel(Model):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]
        
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            self.valid_state()
            score = self.forward(x)
            pred = softmax(score.flatten())

            sampled = np.random.choice(len(pred), size=1, p=pred)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
    

    def get_rnn_state(self):
        states = OrderedDict()
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if repr(layer) == "LSTMLayer":
                states[layer_name] = OrderedDict()
                states[layer_name]["h"] = deepcopy(layer.h)
                states[layer_name]["c"] = deepcopy(layer.c)
                
            elif repr(layer) == "RNNLayer":
                states[layer_name] = OrderedDict()
                states[layer_name]["h"] = deepcopy(layer.h)
                
        return states
    
    
    def set_rnn_state(self, states):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if repr(layer) == "LSTMLayer":
                layer.h = deepcopy(states[layer_name]["h"])
                layer.c = deepcopy(states[layer_name]["c"])
                
            elif repr(layer) == "RNNLayer":
                layer.h = deepcopy(states[layer_name]["h"])
        
                
                