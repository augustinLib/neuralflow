from simpleDL.gpu import *

class DataLoader:
    def __init__(self, x, y, batch_size=1, random_sampling = False):
        """
        Build dataloader that can iterate with batch size in dataset (for supervised task)

        Parameters
        ----------
        x : data corresponding to feature

        y : data corresponding to label

        batch_size (int) : batch size. Default: 1

        random_sampling (bool) : Whether for random sampling from dataset. Default: False

        """
        self.current = 0
        self.x = np.array(x)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.batch_mask = np.arange(self.x.shape[0])
        if random_sampling:
            np.random.shuffle(self.batch_mask)


    def __iter__(self):
        return self


    def __next__(self):
        if self.current + self.batch_size <= len(self.x):
            x = self.x[self.batch_mask[self.current:self.current + self.batch_size]]
            y = self.y[self.batch_mask[self.current:self.current + self.batch_size]]
            self.current += self.batch_size
            return x, y

        elif self.current >= len(self.batch_mask):
            # for next epoch
            self.current = 0
            raise StopIteration
        
        else:
            x = self.x[self.batch_mask[self.current:]]
            y = self.y[self.batch_mask[self.current:]]
            self.current += self.batch_size
            return x, y

        
    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return int(len(self.x) / self.batch_size)

    
    def dataset_len(self):
        return len(self.x)
    
    
class LanguageModelingDataLoader():
    def __init__(self, corpus, batch_size = 1, time_size = 10):
        """
        Build dataloader that can iterate with batch size in dataset (for Language Modeling)

        Parameters
        ----------
        corpus : data corresponding to feature

        batch_size (int) : batch size. Default: 1

        """
        self.current_iter = 0
        self.corpus = np.array(corpus)
        self.x_temp = np.array(corpus[:-1])
        self.y_temp = np.array(corpus[1:])
        self.corpus_size = len(corpus)
        self.data_size = len(self.x_temp)
        self.batch_size = batch_size
        self.time_size = time_size
        self.jump = (self.corpus_size - 1) // self.batch_size
        # offset : 각각의 mini-batch의 시작 요소의 위치를 담음
        self.offset = [i * self.jump for i in range(self.batch_size)]
        self.max_iteration = self.corpus_size // (self.batch_size * self.time_size)
        self.time_index = 0
        

    def __iter__(self):
        return self


    def __next__(self):
        if self.current_iter < self.max_iteration-1:
            x = np.empty((self.batch_size, self.time_size), dtype="i")
            y = np.empty((self.batch_size, self.time_size), dtype="i")
            for t in range(self.time_size):
                for i, offset in enumerate(self.offset):
                    x[i, t] = self.x_temp[(offset + self.time_index) % self.data_size]
                    y[i, t] = self.y_temp[(offset + self.time_index) % self.data_size]
                self.time_index += 1
            self.current_iter += 1
            return x, y

        elif self.current_iter >= self.max_iteration-1:
            # for next epoch
            self.current_iter = 0
            raise StopIteration

        
    def __getitem__(self, index):
        return self.x_temp[index], self.y_temp[index]


    def __len__(self):
        return self.max_iteration

    
    def dataset_len(self):
        return self.corpus_size
        