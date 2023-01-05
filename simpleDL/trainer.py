from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np


class BaseTrainer():
    def __init__(self, n_epochs= 10, init_lr=0.01):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.metric = OrderedDict()
        self.train_loss_list = np.array([])
        self.valid_loss_list = np.array([])


    def __repr__(self) -> str:
        return "Trainer"



class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, critic, optimizer, n_epochs= 10, init_lr=0.01, is_validate = True):
        super().__init__(n_epochs, init_lr)
        self.model = model
        self.critic = critic
        self.optimizer = optimizer
        self.is_validate = is_validate

        self.train_accuracy_list = np.array([])
        self.valid_accuracy_list = np.array([])


    def _forward(self, x, y):
        pred = self.model(x)
        loss = self.critic(pred, y)

        return loss

    
    def _backward(self):
        self.model._backward(self.critic)
        grad = self.model.gradient()
        return grad

    
    def _update(self):
        self.optimizer.update(self.model)
    

    def train(self, train_dataloader, valid_dataloader = None):
        for epoch in range(self.n_epochs):
            print(f"epoch {epoch+1}")
            tmp_train_loss = np.array([])
            train_correct_num = 0
            y_num = 0
            for x, y in tqdm(train_dataloader):
                train_loss = self._forward(x, y)
                tmp_train_loss = np.append(tmp_train_loss, train_loss)
                pred = np.argmax(self.critic.pred, axis=1)

                if y.ndim != 1:
                    y = np.argmax(y, axis=1)
                train_correct_num += np.sum(pred==y)
                y_num += len(y)

                print(f"train loss : {train_loss}    train accuarcy : {train_correct_num/y_num*100}\r", end="")

                _ = self._backward()
                self._update()
            
                
            epoch_train_loss = np.sum(tmp_train_loss)
            self.train_loss_list = np.append(self.train_loss_list, epoch_train_loss)
            dataset_len = train_dataloader.dataset_len()
            epoch_train_accuracy = train_correct_num/float(dataset_len) * 100
            self.train_accuracy_list = np.append(self.train_accuracy_list, epoch_train_accuracy)
            print()
            print(f"epoch {epoch+1} -- train loss : {epoch_train_loss}    train accuarcy : {epoch_train_accuracy}")
            
            
            if valid_dataloader is not None:
                self._validate(valid_dataloader, epoch)
            else:
                print("--------------------------------")    




    def _validate(self, valid_dataloader, epoch):
        tmp_valid_loss = np.array([])
        valid_correct_num = 0
        y_num = 0
        for x, y in tqdm(valid_dataloader):
            valid_loss = self._forward(x, y)
            tmp_valid_loss = np.append(tmp_valid_loss, valid_loss)
            pred = np.argmax(self.critic.pred, axis=1)
            if y.ndim != 1:
                y = np.argmax(y, axis=1)
            valid_correct_num += np.sum(pred==y)
            y_num += len(y)

            print(f"valid loss : {valid_loss}    valid accuarcy : {valid_correct_num/y_num*100}\r", end="")

        epoch_valid_loss = np.sum(tmp_valid_loss)
        self.valid_loss_list = np.append(self.valid_loss_list, epoch_valid_loss)
        dataset_len = valid_dataloader.dataset_len()
        epoch_valid_accuracy = valid_correct_num/float(dataset_len) * 100
        self.valid_accuracy_list = np.append(self.valid_accuracy_list, epoch_valid_accuracy)
        print()
        print(f"epoch {epoch+1} -- valid loss : {epoch_valid_loss}    valid accuarcy : {epoch_valid_accuracy}")
        print("--------------------------------")    


    def show_error_graph(self):
        plt.title("train/valid loss per epoch")
        plt.plot(self.train_loss_list, "-r" ,label = 'train error')
        plt.plot(self.valid_loss_list, "-b" ,label = 'valid error')

        plt.legend(loc="upper right")

        plt.grid()
        plt.show()


    def show_accuracy_graph(self):
        plt.title("train/valid accuarcy per epoch")
        plt.plot(self.train_accuracy_list, "-r" ,label = 'train accuracy')
        plt.plot(self.valid_accuracy_list, "-b" ,label = 'valid accuracy')

        plt.legend(loc="lower right")

        plt.grid()
        plt.show()



    def add_metric(self, **metric):
        print(f"metic {metric.keys()} added")
        self.metric = dict(self.metric, metric)