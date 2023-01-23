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
                
                
class Encoder(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, dropout = None, *layers):
        super().__init__(*layers)
        if len(layers) == 0:
            self.add_layer(EmbeddingLayer(vocab_size=vocab_size, hidden_size=word_vec_size))
            for i in range(n_layers):
                self.add_layer(LSTMLayer(word_vec_size, hidden_size))
                if dropout != None:
                    self.add_layer(Dropout(dropout_ratio=dropout))
            
        self.hidden_state = None
    
    
    def _forward(self, x):
        hidden_state = self.forward(x)
        self.hidden_state = hidden_state
        
        return hidden_state[:, -1, :]
    
    
    def _backward(self, dh):
        dh_t = np.zeros_like(self.hs)
        dh_t[:, -1, :] = dh
        
        self.backward(dh_t)
        

class Decoder(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, dropout = None, *layers):
        super().__init__(*layers)
        if len(layers) == 0:
            self.add_layer(EmbeddingLayer(vocab_size=vocab_size, hidden_size=word_vec_size))
            for i in range(n_layers):
                self.add_layer(LSTMLayer(word_vec_size, hidden_size))
                if dropout != None:
                    self.add_layer(Dropout(dropout_ratio=dropout))
                    
            self.add_layer(DenseLayer(hidden_size, vocab_size))
        
    
    def _forward(self, x, h):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                layer.load_state(h)
                break
        
        result = self.forward(x)
        
        return result
    
    
    def _backward(self, dout):
        self.backward(dout)
        dh = 0
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                dh = layer.dh
                break
            
        return dh
        
    
    def generate(self, h, start_id, sample_size):
        sampled_result = []
        sample_id = start_id
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                layer.load_state(h)
        
        for i in range(sample_size):
            x = np.array(sample_id).reshape((1,1))
            result = self._forward(x)
            
        sample_id = np.argmax(result.flatten())
        sampled_result.append(int(sample_id))
        
        return sampled_result
    
    
class Seq2Seq(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, dropout = None):
        super().__init__(())
        self.encoder = Encoder(vocab_size=vocab_size,
                               word_vec_size=word_vec_size,
                               hidden_size=hidden_size,
                               n_layers=n_layers,
                               dropout=dropout)
        
        
        self.decoder = Decoder(vocab_size=vocab_size,
                               word_vec_size=word_vec_size,
                               hidden_size=hidden_size,
                               n_layers=n_layers,
                               dropout=dropout)
        
        self.layers = self.encoder.layers + self.decoder.layers
        
        for layer in self.layers:
            self.network[f"{repr(layer)}{self.count_dict[repr(layer)]}"] = layer
            self.sequence.append(f"{repr(layer)}{self.count_dict[repr(layer)]}")
            self.count_dict[repr(layer)] += 1
            
            
    def __call__(self, *args):
        result = self.forward(*args)
        return result
        
    
    def forward(self, x, decoder_x):
        h = self.encoder._forward(x)
        result = self.decoder._forward(decoder_x, h)

        return result
    
    
    def backward(self, loss):
        dout = loss._backward()
        dh = self.decoder._backward(dout)
        self.encoder._backward(dh)


    def generate(self, x, start_id, sample_size):
        h = self.encoder._forward(x)
        sampled = self.decoder.generate(h,
                                        start_id=start_id,
                                        sample_size=sample_size)
        
        return sampled
        
    


        
        
    
    
                

        
                
                