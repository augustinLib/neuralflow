from collections import Counter
from neuralflow.gpu import *
from neuralflow.function import *
from neuralflow.model import *


class BaseFunction():
    def __init__(self):
        self.differentiable = False
        self.changeability = False
        self.mixed_precision = False


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def _forward(self, x):
        pass


    def __repr__(self) -> str:
        return "Function"
    
    
    def _mixed_precision_training(self):
        self.mixed_precision = True


class Step(BaseFunction):
    def __init__(self):
        super().__init__()

    def _forward(self, x):
        return step_function(x)


class Identity(BaseFunction):
    def __init__(self):
        super().__init__()

    def _forward(self, x):
        return identity(x)


class Sigmoid(BaseFunction):
    def __init__(self):
        super().__init__()
        self.differentiable = False
        self.out = None
        

    def _forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out


    def _backward(self, input):
        result = input * self.out * (1.0-self.out)

        return result
    
    
class Tanh(BaseFunction):
    def __init__(self):
        super().__init__()
        self.differentiable = False
        self.out = None
        

    def _forward(self, x):
        out = np.tanh(x)
        self.out = out

        return out


    def _backward(self, input):
        result = input * (1 - self.out ** 2)

        return result

    
class Softmax(BaseFunction):
    def __init__(self):
        super().__init__()
        self.out = None

    def _forward(self, x):
        self.out = softmax(x)
        return self.out

    def _backward(self, input):
        dx = self.out * input
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class ReLU(BaseFunction):
    def __init__(self):
        self.differentiable = False
        self.mask = None
        super().__init__()

    def _forward(self, x):
        self.mask = (x <= 0)
        result = x.copy()
        result[self.mask] = 0
        
        return result

    def _backward(self, input):
        input[self.mask] = 0
        result = input

        return result


class CrossEntropyLoss():
    def __init__(self, loss_scaling = None):
        self.loss = None
        self.pred = None
        self.true = None
        self.cache = None
        self.ignore_label = -1
        self.mixed_precision = False
        self.scaling_factor = loss_scaling
        
    def __repr__(self):
        return "Function"


    def __call__(self, *arg):
        result = self._forward(*arg)
        return result


    def _forward(self, pred, true):
        if pred.ndim == 3:
            batch_size, n_timestep, vocab_size = pred.shape
            if true.ndim == 3:
                true = true.argmax(axis=2)
            
            mask = (true != self.ignore_label)
            
            self.pred = pred.reshape(batch_size * n_timestep, vocab_size)
            self.true = true.reshape(batch_size * n_timestep)
            mask = mask.reshape(batch_size * n_timestep)
            pred_sentence = softmax(self.pred)
            ls = np.log(pred_sentence[np.arange(batch_size * n_timestep), self.true])
            ls *= mask
            self.loss = -np.sum(ls)
            self.loss /= mask.sum()
            self.cache = (self.true, pred_sentence, mask, (batch_size, n_timestep, vocab_size))
            
        else:
            self.pred = softmax(pred)
            self.true = true
            self.loss = cross_entropy_error(self.pred, self.true)
            
        return self.loss

    def _backward(self):
        if self.cache is not None:
            true, pred_sentence, mask, (batch_size, n_timestep, vocab_size) = self.cache

            dx = pred_sentence
            dx[np.arange(batch_size * n_timestep), true] -= 1
            dx *= 1
            dx /= mask.sum()
            dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

            dx = dx.reshape((batch_size, n_timestep, vocab_size))

            return dx
        
        else:
            batch_size = self.true.shape[0]
            if self.true.size == self.pred.size:
                dx = (self.pred - self.true) / batch_size
            else:
                dx = self.pred.copy()
                dx[np.arange(batch_size), self.true] -= 1
                dx = dx / batch_size


            if self.scaling_factor != None:
                dx = dx * self.scaling_factor

            return dx


class BinaryCrossEntropyLoss():
    def __init__(self, loss_scaling = None):
        self.loss = None
        self.pred = None
        self.true = None
        self.mixed_precision = False
        self.scaling_factor = loss_scaling

    def __repr__(self) -> str:
        return "Function"

    def __call__(self, *arg):
        result = self._forward(*arg)
        return result

    def _forward(self, pred, true):
        self.pred = sigmoid(pred)
        self.true = true
        self.loss = cross_entropy_error(np.c_[1- self.pred, self.pred], self.true)

        return self.loss

    def _backward(self):
        batch_size = self.true.shape[0]
        if self.true.size == self.pred.size:
            dx = (self.pred - self.true) / batch_size
        else:
            dx = self.pred.copy()
            dx[np.arange(batch_size), self.true] -= 1
            dx = dx / batch_size

        return dx


class Sampler():
    def __init__(self, corpus, power = 0.75, sample_size = 2 ):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def negative_sampling(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


