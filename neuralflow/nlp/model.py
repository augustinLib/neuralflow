from copy import deepcopy
from collections import OrderedDict
from neuralflow.gpu import *
from neuralflow.function import *
from neuralflow.function_class import *
from neuralflow.model import *




# class NegativeSamplingLoss(Model):
#     def __init__(self, corpus, power=0.75, sample_size=5):
#         super().__init__()
#         self.sample_size = sample_size
#         self.sampler = Sampler(corpus, power, sample_size)
#         for i in range(sample_size+1):
#             self.add    
#         self.loss_layers = [BinaryCrossEntropyLoss() for _ in range(sample_size + 1)]
#         self.embed_dot_layers = [EmbeddingLayer(vocab_size, hidden_size) for _ in range(sample_size + 1)]

#         self.params, self.grads = [], []
#         for layer in self.embed_dot_layers:
#             self.params += layer.params
#             self.grads += layer.grads

#     def forward(self, h, target):
#         batch_size = target.shape[0]
#         negative_sample = self.sampler.get_negative_sample(target)

#         # 긍정적 예 순전파
#         score = self.embed_dot_layers[0].forward(h, target)
#         correct_label = np.ones(batch_size, dtype=np.int32)
#         loss = self.loss_layers[0].forward(score, correct_label)

#         # 부정적 예 순전파
#         negative_label = np.zeros(batch_size, dtype=np.int32)
#         for i in range(self.sample_size):
#             negative_target = negative_sample[:, i]
#             score = self.embed_dot_layers[1 + i].forward(h, negative_target)
#             loss += self.loss_layers[1 + i].forward(score, negative_label)

#         return loss

#     def backward(self, dout=1):
#         dh = 0
#         for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
#             dscore = l0.backward(dout)
#             dh += l1.backward(dscore)

#         return dh


class LanguageModel(Model):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]
        
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            self.eval_state()
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
                self.add_layer(LSTMLayer(word_vec_size, hidden_size, stateful=False))
                if dropout != None:
                    self.add_layer(Dropout(dropout_ratio=dropout))
            
        self.hidden_state = None
    
    
    def _forward(self, x):
        hidden_state = self.forward(x)
        self.hidden_state = hidden_state
        
        return hidden_state[:, -1, :]
    
    
    def _backward(self, dh_t):
        dhidden_state = np.zeros_like(self.hidden_state)
        dhidden_state[:, -1, :] = dh_t

        last_layer_name = self.sequence[-1]
        last_layer = self.network[last_layer_name]
        result = last_layer._backward(dhidden_state)
        
        for layer_name in reversed(self.sequence[:-1]):
            layer = self.network[layer_name]
            result = layer._backward(result)
        
        if self.tying_weight:
            self.tying_backward()
        

class Decoder(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, peeky=False, dropout = None, *layers):
        super().__init__(*layers)
        if len(layers) == 0:
            if peeky:
                self.add_layer(EmbeddingLayer(vocab_size=vocab_size, hidden_size=word_vec_size, initialize=100))
                for i in range(n_layers):
                    self.add_layer(LSTMLayer(word_vec_size+hidden_size, hidden_size))
                    if dropout != None:
                        self.add_layer(Dropout(dropout_ratio=dropout))
                        
                self.add_layer(DenseLayer(hidden_size+hidden_size, vocab_size))
                
            else:
                self.add_layer(EmbeddingLayer(vocab_size=vocab_size, hidden_size=word_vec_size, initialize=100))
                for i in range(n_layers):
                    self.add_layer(LSTMLayer(word_vec_size, hidden_size))
                    if dropout != None:
                        self.add_layer(Dropout(dropout_ratio=dropout))
                        
                self.add_layer(DenseLayer(hidden_size, vocab_size))

        self.hidden_size=hidden_size
        self.peeky=peeky
        
    def _forward(self, x, h):
        if self.tying_weight:
            self.tying_forward()
            
        if self.peeky:
            batch_size, n_timestep = x.shape
            
            for layer_name in self.sequence:
                layer = self.network[layer_name]
                if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                    layer.load_state(h)
                    break
            
            first_emb_name = self.sequence[0]
            first_emb = self.network[first_emb_name]
            result = first_emb(x)
            
            hidden_state = np.repeat(h, n_timestep, axis=0).reshape(batch_size, n_timestep, self.hidden_size)
            
            for layer_name in self.sequence[1:-1]:
                layer = self.network[layer_name]
                if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                    result = np.concatenate((hidden_state, result), axis=2)
                result = layer(result)
                
            result = np.concatenate((hidden_state, result), axis=2)
            final_dense_name = self.sequence[-1]
            final_dense = self.network[final_dense_name]
            result = final_dense(result)
                    
        else:
            for layer_name in self.sequence:
                layer = self.network[layer_name]
                if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                    layer.load_state(h)
                    break
                
            input = x

            for layer_name in self.sequence:
                layer = self.network[layer_name]
                y = layer(input)
                input = y
    
            result = input
        
        return result
    
    
    def _backward(self, loss):
        if self.peeky:
            final_dense_name = self.sequence[-1]
            final_dense = self.network[final_dense_name]
            dout = final_dense._backward(loss)
            dout, dhs0 = dout[:, :, self.hidden_size:], dout[:, :, :self.hidden_size]
            
            for layer_name in reversed(self.sequence[1:-1]):
                layer = self.network[layer_name]
                dout = layer._backward(dout)  
                if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                    dembed, dhs1 = dout[:, :, self.hidden_size:], dout[:, :, :self.hidden_size]    
                    
            first_emb_name = self.sequence[0]
            first_emb = self.network[first_emb_name]
            first_emb._backward(dembed)
            dhs = dhs0 + dhs1
            
            first_lstm_dh = None
            for layer_name in self.sequence:
                layer = self.network[layer_name]
                if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                    first_lstm_dh = layer.dh
                    break
                
            dh = first_lstm_dh + np.sum(dhs, axis=1)
                        
        else:
            last_layer_name = self.sequence[-1]
            last_layer = self.network[last_layer_name]
            result = last_layer._backward(loss)

            for layer_name in reversed(self.sequence[:-1]):
                layer = self.network[layer_name]
                result = layer._backward(result)

            dh = 0
            for layer_name in self.sequence:
                layer = self.network[layer_name]
                if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                    dh = layer.dh
                    break
                
        if self.tying_weight:
            self.tying_backward()
                
        return dh
                

    def generate(self, h, start_id, sample_size):
        sampled_result = []
        sample_id = start_id
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                layer.load_state(h)
                break
            
        if self.peeky:
            peeky_h = h.reshape(1,1,self.hidden_size)
            for i in range(sample_size):
                x = np.array(sample_id).reshape((1,1))
                result = x
                
                first_emb_name = self.sequence[0]
                first_emb = self.network[first_emb_name]
                result = first_emb(x)

                for layer_name in self.sequence[1:-1]:
                    layer = self.network[layer_name]
                    if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                        result = np.concatenate((peeky_h, result), axis=2)
                    result = layer(result)

                result = np.concatenate((peeky_h, result), axis=2)
                final_dense_name = self.sequence[-1]
                final_dense = self.network[final_dense_name]
                result = final_dense(result)

                sample_id = np.argmax(result.flatten())
                sampled_result.append(int(sample_id))
            
        else:
            for i in range(sample_size):
                x = np.array(sample_id).reshape((1,1))
                result = x
                for layer_name in self.sequence:
                    layer = self.network[layer_name]
                    result = layer._forward(result)

                sample_id = np.argmax(result.flatten())
                sampled_result.append(int(sample_id))
        
        return sampled_result
    
    
    
class Seq2Seq(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, peeky = False, dropout = None):
        super().__init__(())
        self.encoder = Encoder(vocab_size=vocab_size,
                               word_vec_size=word_vec_size,
                               hidden_size=hidden_size,
                               n_layers=n_layers,
                               dropout=dropout)
        
        if peeky:
            self.decoder = Decoder(vocab_size=vocab_size,
                                   word_vec_size=word_vec_size,
                                   hidden_size=hidden_size,
                                   n_layers=n_layers,
                                   peeky=True,
                                   dropout=dropout)
            
        else:
            self.decoder = Decoder(vocab_size=vocab_size,
                                   word_vec_size=word_vec_size,
                                   hidden_size=hidden_size,
                                   n_layers=n_layers,
                                   dropout=dropout)
        
        self.network = OrderedDict()
        self.count_dict = OrderedDict()
        self.layers = self.encoder.layers + self.decoder.layers
        self.sequence = []
        temp_repr_list = []
        
        for layer in self.layers:
            temp_repr_list.append(repr(layer))
        repr_set = set(temp_repr_list)

        #initialize count_dict
        for rep in repr_set:
            self.count_dict[rep] = 1     

        for layer in self.layers:
            self.network[f"{repr(layer)}{self.count_dict[repr(layer)]}"] = layer
            self.sequence.append(f"{repr(layer)}{self.count_dict[repr(layer)]}")
            self.count_dict[repr(layer)] += 1
            
            
    def __repr__(self) -> str:
        return "Seq2Seq"
            
            
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
    
    
    def weight_tying(self):
        self.decoder.weight_tying()


class WeightSum(BaseFunction):
    def __init__(self):
        super().__init__()
        self.differentiable = False
        self.changeability = False
        self.mixed_precision = False

        self.cache = None

    def _forward(self, hidden_state, weigth):
        batch_size, n_timestep, hidden_size = hidden_state.shape

        # with broadcast
        # repeated_attention : (batch_size, n_timestep, hidden_state)
        repeated_attention = weigth.reshape(batch_size, n_timestep, 1)
        
        # hidden_state : (batch_size, n_timestep, hidden_state)
        t = hidden_state * repeated_attention
        
        # hidden_size의 요소들 하나로 합쳐서 context생성
        # context : (batch_size, hidden_state)
        context = np.sum(t, axis=1)

        self.cache = (hidden_state, repeated_attention)
        return context

    def _backward(self, dcontext):
        hidden_state, repeated_attention = self.cache
        batch_size, n_timestep, hidden_size = hidden_state.shape
        
        dt = dcontext.reshape(batch_size, 1, hidden_size).repeat(n_timestep, axis=1)
        
        drepeated_attention = dt * hidden_state
        dhidden_state = dt * repeated_attention
        # repeat된거 합하기
        dweight = np.sum(drepeated_attention, axis=2)

        return dhidden_state, dweight


class AttentionWeight(BaseFunction):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
        self.cache = None
        self.differentiable = False
        self.changeability = False
        self.mixed_precision = False

    def _forward(self, hidden_state, h_t):
        batch_size, n_timestep, hidden_size = hidden_state.shape

        # h_t : 하나의 timestep의 hidden_state
        # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        repeated_h = h_t.reshape(batch_size, 1, hidden_size)
        # t : (batch_size, n_timestep ,hidden_size)
        t = hidden_state * repeated_h
        
        # score : (batch_size, n_timestep)
        score = np.sum(t, axis=2)
        # attention : (batch_size, n_timestep)
        attention = self.softmax._forward(score)

        self.cache = (hidden_state, repeated_h)
        return attention

    def _backward(self, dattention):
        hidden_state, repeated_h = self.cache
        batch_size, n_timestep, hidden_size = hidden_state.shape

        # score : (batch_size, n_timestep)
        dscore = self.softmax._backward(dattention)
        
        # t : (batch_size, n_timestep ,hidden_size)
        dt = dscore.reshape(batch_size, n_timestep, 1).repeat(hidden_size, axis=2)
        dhidden_state = dt * repeated_h
        drepeated_h = dt * hidden_state
        # repeat된거 합하기
        dh_t = np.sum(drepeated_h, axis=1)

        return dhidden_state, dh_t


class Attention(BaseFunction):
    def __init__(self):
        super().__init__()
        self.differentiable = False
        self.changeability = False
        self.mixed_precision = False

        self.weight_sum = WeightSum()
        self.attention_weight = AttentionWeight()
        self.attention = None
        
        
    def _forward(self, hidden_state, h_t):
        attention = self.attention_weight._forward(hidden_state, h_t)
        result = self.weight_sum._forward(hidden_state, attention)
        self.attention = attention
        return result
    
    
    def _backward(self, dout):
        dhidden_state_0, dattention = self.weight_sum._backward(dout)
        dhidden_state_1, dh_t = self.attention_weight._backward(dattention)
        dhidden_state = dhidden_state_0 + dhidden_state_1
        return dhidden_state, dh_t
    
    
class AttentionLayer(BaseLayer):
    def __init__(self):
        super().__init__()
        self.differentiable = False
        self.changeability = False
        self.mixed_precision = False
        self.attention_cells = None
        self.attention_weights = None


    def __call__(self, *args):
        result = self._forward(*args)
        return result
    
    
    def __repr__(self) -> str:
        return "AttentionLayer"
    

    def _forward(self, encoder_hidden_state, decoder_hidden_state):
        batch_size, n_timestep, hidden_size = decoder_hidden_state.shape
        result = np.empty_like(decoder_hidden_state).astype(np.float32)
        self.attention_cells = []
        self.attention_weights = []

        for timestep in range(n_timestep):
            cell = Attention()
            result[:, timestep, :] =  cell._forward(encoder_hidden_state, decoder_hidden_state[:,timestep,:])
            self.attention_cells.append(cell)
            self.attention_weights = np.append(self.attention_weights, cell.attention)

        return result

    def _backward(self, dout):
        batch_size, n_timestep, hidden_size = dout.shape
        encoder_dhidden_state = 0
        decoder_dhidden_state = np.empty_like(dout).astype(np.float32)

        for timestep in range(n_timestep):
            cell = self.attention_cells[timestep]
            dhidden_state, dh_t = cell._backward(dout[:, timestep, :])
            encoder_dhidden_state += dhidden_state
            decoder_dhidden_state[:,timestep,:] = dh_t

        return encoder_dhidden_state, decoder_dhidden_state
    
    
class AttentionEncoder(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, dropout = None, *layers):
        super().__init__(*layers)
        if len(layers) == 0:
            self.add_layer(EmbeddingLayer(vocab_size=vocab_size, hidden_size=word_vec_size))
            for i in range(n_layers):
                self.add_layer(LSTMLayer(word_vec_size, hidden_size, stateful=False))
                if dropout != None:
                    self.add_layer(Dropout(dropout_ratio=dropout))
                    
    
    def _forward(self, x):
        hidden_state = self.forward(x)
        
        return hidden_state
    
    
    def _backward(self, dhidden_state):
        result = dhidden_state
        for layer_name in reversed(self.sequence):
            layer = self.network[layer_name]
            result = layer._backward(result)
        
        if self.tying_weight:
            self.tying_backward()
            
        return result
            
    
class AttentionDecoder(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, dropout = None, *layers):
        super().__init__(*layers)
        if len(layers) == 0:
            self.add_layer(EmbeddingLayer(vocab_size=vocab_size, hidden_size=word_vec_size, initialize=100))
            for i in range(n_layers):
                self.add_layer(LSTMLayer(word_vec_size, hidden_size))
                if dropout != None:
                    self.add_layer(Dropout(dropout_ratio=dropout))
                    
            self.add_layer(AttentionLayer())
            self.add_layer(DenseLayer(hidden_size*2, vocab_size))
        
        
    def _forward(self, x, encoder_hidden_state):
        h = encoder_hidden_state[:,-1]
        layer = self.network["LSTMLayer1"]
        layer.load_state(deepcopy(h))

        if self.tying_weight:
            self.tying_forward()
            
        input = x
            
        # LSTM까지만 진행
        for layer_name in self.sequence[:-2]:
            layer = self.network[layer_name]
            y = layer._forward(input)
            input = y

        decoder_hidden_state = input
        
        attention = self.network["AttentionLayer1"]
        context = attention._forward(encoder_hidden_state, decoder_hidden_state)
        
        final_dense = self.network["DenseLayer1"]
        
        result = np.concatenate((context, decoder_hidden_state), axis=2).astype(np.float32)
        result = final_dense._forward(result)
        
        return result
    
    
    def _backward(self, dout):
        # dense layer
        final_dense = self.network["DenseLayer1"]
        dout = final_dense._backward(dout)
        
        batch_size, n_timestep, hidden_size_double = dout.shape
        hidden_size = hidden_size_double // 2
        
        # attention layer
        attention = self.network["AttentionLayer1"]
        
        dcontext, ddecoder_hidden_state_0 = dout[:,:,:hidden_size], dout[:,:,hidden_size:]
        dencoder_hidden_state, ddecoder_hidden_state_1 = attention._backward(dcontext)
        ddecoder_hidden_state = ddecoder_hidden_state_0 + ddecoder_hidden_state_1
        
        # lstm부터 끝(embedding)까지
        dout = ddecoder_hidden_state
        for layer_name in reversed(self.sequence[:-2]):
            layer = self.network[layer_name]
            dout = layer._backward(dout)  
        
        # 이후 encoder로 전달받은 gradient 합산해서 전달
        first_lstm_dh = self.network["LSTMLayer1"].dh
        dencoder_hidden_state[:, -1] += first_lstm_dh

        return dencoder_hidden_state
                

    def generate(self, encoder_hidden_state, start_id, sample_size):
        sampled_result = []
        sample_id = start_id
        h = encoder_hidden_state[:,-1]
        layer = self.network["LSTMLayer1"]
        layer.load_state(deepcopy(h))
            
        if self.tying_weight:
            self.tying_forward()
            
        for _ in range(sample_size):
            input = np.array([sample_id]).reshape((1, 1))
            # LSTM까지만 진행
            for layer_name in self.sequence[:-2]:
                layer = self.network[layer_name]
                y = layer._forward(input)
                input = y

            decoder_hidden_state = input

            attention = self.network["AttentionLayer1"]
            context = attention._forward(encoder_hidden_state, decoder_hidden_state)

            final_dense = self.network["DenseLayer1"]

            result = np.concatenate((context, decoder_hidden_state), axis=2).astype(np.float32)
            result = final_dense._forward(result)

            sample_id = np.argmax(result.flatten())
            sampled_result.append(sample_id)
        
        return sampled_result
    
    
class AttentionSeq2Seq(Model):
    def __init__(self, vocab_size=1000, word_vec_size=128, hidden_size=128, n_layers=1, dropout = None):
        super().__init__(())
        self.encoder = AttentionEncoder(vocab_size=vocab_size,
                               word_vec_size=word_vec_size,
                               hidden_size=hidden_size,
                               n_layers=n_layers,
                               dropout=dropout)
        

        self.decoder = AttentionDecoder(vocab_size=vocab_size,
                                word_vec_size=word_vec_size,
                                hidden_size=hidden_size,
                                n_layers=n_layers,
                                dropout=dropout)

        self.network = OrderedDict()
        self.count_dict = OrderedDict()
        self.layers = self.encoder.layers + self.decoder.layers
        self.sequence = []
        temp_repr_list = []
        
        for layer in self.layers:
            temp_repr_list.append(repr(layer))
        repr_set = set(temp_repr_list)

        #initialize count_dict
        for rep in repr_set:
            self.count_dict[rep] = 1     

        for layer in self.layers:
            self.network[f"{repr(layer)}{self.count_dict[repr(layer)]}"] = layer
            self.sequence.append(f"{repr(layer)}{self.count_dict[repr(layer)]}")
            self.count_dict[repr(layer)] += 1
            
            
    def __repr__(self) -> str:
        return "Seq2Seq"
            
            
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