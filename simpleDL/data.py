import numpy as np

class DataLoader:
    def __init__(self, x, y, batch_size=1, random_sampling = False) -> None:
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