import sys
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
from collections import OrderedDict
from neuralflow.gpu import *
from neuralflow.utils import *


class BaseTrainer():
    def __init__(self, n_epochs= 10, init_lr=0.01):
        """
        Base class for Trainer

        Parameters
        ----------
        n_epochs (int) : number of epochs. Default: 10

        init_lr (int) : initial learning rate. Default: 0.01

        """
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.metric = OrderedDict()
        self.train_loss_list = np.array([])
        self.valid_loss_list = np.array([])


    def __repr__(self):
        return "Trainer"


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, critic, optimizer, n_epochs= 10, init_lr=0.01):
        """
        Trainer for Classification

        Parameters
        ----------
        model : model for classification task

        critic : loss function class

        optimizer : optimizer for model optimizing

        n_epochs (int) : number of epochs. Default: 10

        init_lr (int) : initial learning rate. Default: 0.01

        """
        super().__init__(n_epochs, init_lr)
        self.model = model
        self.critic = critic
        self.optimizer = optimizer

        self.train_accuracy_list = np.array([])
        self.valid_accuracy_list = np.array([])


    def _forward(self, x, y):
        pred = self.model(x)
        loss = self.critic(pred, y)

        return loss

    
    def _backward(self):
        self.model.backward(self.critic)
        
    
    def _update(self):
        self.optimizer.update(self.model)
    

    def train(self, train_dataloader, valid_dataloader = None):
        """
        Train model with train/valid data

        Parameters
        ----------
        train_dataloader (iterable) : Train dataloader that can iterate with batch size in train dataset

        valid_dataloader (iterable) : Valid dataloader that can iterate with batch size in valid dataset

        """
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

                self._backward()
                self._update()
            
                
            epoch_train_loss = np.sum(tmp_train_loss) / train_dataloader.batch_size
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

        epoch_valid_loss = np.sum(tmp_valid_loss) / valid_dataloader.batch_size
        self.valid_loss_list = np.append(self.valid_loss_list, epoch_valid_loss)
        dataset_len = valid_dataloader.dataset_len()
        epoch_valid_accuracy = valid_correct_num/float(dataset_len) * 100
        self.valid_accuracy_list = np.append(self.valid_accuracy_list, epoch_valid_accuracy)
        print()
        print(f"epoch {epoch+1} -- valid loss : {epoch_valid_loss}    valid accuarcy : {epoch_valid_accuracy}")
        print("--------------------------------")    

        
    def show_error_graph(self, valid = False):
        """
        Visualize training/validation error
        """
        if valid:
            plt.plot(to_cpu(self.valid_loss_list), "-b" ,label = 'valid error')
            plt.title("train/valid loss per epoch")
        else:
            plt.title("train loss per epoch")
            
        plt.plot(to_cpu(self.train_loss_list), "-r" ,label = 'train error')

        plt.legend(loc="upper right")

        plt.grid()
        plt.show()


    def show_accuracy_graph(self, valid = False):
        """
        Visualize training/validation accuracy
        """
        if valid:
            plt.plot(to_cpu(self.valid_accuracy_list), "-b" ,label = 'valid accuracy')
            plt.title("train/valid accuracy per epoch")
        else:
            plt.title("train accuracy per epoch")
            
        plt.plot(to_cpu(self.train_accuracy_list), "-r" ,label = 'train accuracy')

        plt.legend(loc="lower right")

        plt.grid()
        plt.show()


    def add_metric(self, **metric):
        print(f"metic {metric.keys()} added")
        self.metric = dict(self.metric, metric)
        
        
class LanguageModelTrainer(BaseTrainer):
    def __init__(self, model, critic, optimizer, n_epochs= 10, init_lr=0.01):
        """
        Trainer for Language modeling

        Parameters
        ----------
        model : model for classification task

        critic : loss function class

        optimizer : optimizer for model optimizing

        n_epochs (int) : number of epochs. Default: 10

        init_lr (int) : initial learning rate. Default: 0.01

        """
        super().__init__(n_epochs, init_lr)
        self.model = model
        self.critic = critic
        self.optimizer = optimizer
        self.train_perplexity_list_iter = np.array([])
        self.valid_perplexity_list_iter = np.array([])
        self.train_perplexity_list = np.array([])
        self.valid_perplexity_list = np.array([])


    def _forward(self, x, y):
        pred = self.model(x)
        loss = self.critic(pred, y)

        return loss

    
    def _backward(self):
        self.model.backward(self.critic)
        
    
    def _update(self):
        self.optimizer.update(self.model)
    

    def train(self, train_dataloader, valid_dataloader = None, max_grad = None, iter_num = 20):
        """
        Train model with train/valid data

        Parameters
        ----------
        train_dataloader (iterable) : Train dataloader that can iterate with batch size in train dataset

        valid_dataloader (iterable) : Valid dataloader that can iterate with batch size in valid dataset

        """
        self.model.train_state()
        for epoch in range(self.n_epochs):
            print(f"epoch {epoch+1}")
            tmp_train_loss = np.array([])
            tmp_train_perplexity = np.array([])
            iter_tmp_train_perplexity = np.array([])
            count = 0
            iter_num = len(train_dataloader)

            
            for x, y in tqdm(train_dataloader):
                count +=1
                # train loss
                train_loss = self._forward(x, y)
                tmp_train_loss = np.append(tmp_train_loss, train_loss)
                # train perplexity
                train_perplexity = np.exp(train_loss)
                tmp_train_perplexity = np.append(tmp_train_perplexity, train_perplexity)
                iter_tmp_train_perplexity = np.append(iter_tmp_train_perplexity, train_perplexity)
                
                if count % iter_num == 0:
                    iter_train_perplexity = np.sum(iter_tmp_train_perplexity) / iter_num
                    iter_tmp_train_perplexity = np.array([])
                    self.train_perplexity_list_iter = np.append(self.train_perplexity_list_iter, iter_train_perplexity)

                print(f"train loss : {train_loss}    train perplexity : {train_perplexity}    iter : {count}/{iter_num}\r", end="")

                self._backward()
                if max_grad is not None:
                    self.clip_grad(max_grad)
                self._update()
                
            
            epoch_train_loss = np.sum(tmp_train_loss) / count
            self.train_loss_list = np.append(self.train_loss_list, epoch_train_loss)
            
            
            epoch_train_perplexity = np.sum(tmp_train_perplexity) / count
            self.train_perplexity_list = np.append(self.train_perplexity_list, epoch_train_perplexity)
            
            print()
            print(f"epoch {epoch+1} -- train loss : {epoch_train_loss}    train perplexity : {epoch_train_perplexity}")
            
            
            if valid_dataloader is not None:
                self._validate(valid_dataloader, epoch)
            else:
                print("--------------------------------")
                print()


    def _validate(self, valid_dataloader, epoch, iter_num = 20):
        self.model.valid_state()
        self.model.reset_rnn_state()
        tmp_valid_loss = np.array([])
        tmp_valid_perplexity = np.array([])
        iter_tmp_valid_perplexity = np.array([])
        count = 0
        for x, y in tqdm(valid_dataloader):
            count +=1
            # valid loss
            valid_loss = self._forward(x, y)
            tmp_valid_loss = np.append(tmp_valid_loss, valid_loss)
            # valid perplexity
            valid_perplexity = np.exp(valid_loss)
            tmp_valid_perplexity = np.append(tmp_valid_perplexity, valid_perplexity)
            iter_tmp_valid_perplexity = np.append(iter_tmp_valid_perplexity, valid_perplexity)
            
            if count % iter_num == 0:
                iter_valid_perplexity = np.sum(iter_tmp_valid_perplexity) / iter_num
                iter_tmp_valid_perplexity = np.array([])
                self.valid_perplexity_list_iter = np.append(self.valid_perplexity_list_iter, iter_valid_perplexity)
            
            print(f"valid loss : {valid_loss}    valid perplexity : {valid_perplexity}\r", end="")
            
        epoch_valid_loss = np.sum(tmp_valid_loss) / count
        self.valid_loss_list = np.append(self.valid_loss_list, epoch_valid_loss)
        
        epoch_valid_perplexity = np.sum(tmp_valid_perplexity) / count
        self.valid_perplexity_list = np.append(self.valid_perplexity_list, epoch_valid_perplexity)
        self.model.reset_rnn_state()
        
        print()
        print(f"epoch {epoch+1} -- valid loss : {epoch_valid_loss}    valid perplexity : {epoch_valid_perplexity}")
        print("--------------------------------")
        print()


    def clip_grad(self, max_norm):
        total_norm = 0
        param_grad_dict = self.optimizer.param_grad_dict
        
        for layer_name in self.model.sequence:
            layer = self.model.network[layer_name]
            
            # only update differentiable layer
            if layer.differentiable:
                grad = layer.get_gradient()
                param_list = list(layer.parameter.keys())
                for param in param_list:
                    total_norm += np.sum(grad[param_grad_dict[param]] ** 2)
                
        total_norm = np.sqrt(total_norm)
        rate = max_norm / (total_norm + 1e-6)
        
        if rate < 1:
            for layer_name in self.model.sequence:
                layer = self.model.network[layer_name]

                # only update differentiable layer
                if layer.differentiable:
                    grad = layer.get_gradient()
                    param_list = list(layer.parameter.keys())
                    for param in param_list:
                        grad[param_grad_dict[param]] *= rate
                


    def show_error_graph(self, valid = False):
        """
        Visualize training/validation error
        """
        if valid:
            plt.plot(to_cpu(self.valid_loss_list), "-b" ,label = 'valid error')
            plt.title("train/valid loss per epoch")
        else:
            plt.title("train loss per epoch")
            
        plt.plot(to_cpu(self.train_loss_list), "-r" ,label = 'train error')

        plt.legend(loc="upper right")

        plt.grid()
        plt.show()


    def show_perplexity_graph(self, show_iter = False, valid = False):
        """
        Visualize training/validation perplexity
        """
        if show_iter:
            plt.ylim(-1, 501)
            
            if valid:
                plt.plot(to_cpu(self.valid_perplexity_list_iter), "-b" ,label = 'valid perplexity')
                plt.title("train/valid perplexity per iteration")
            else:
                plt.title("train perplexity per iteration")

            plt.plot(to_cpu(self.train_perplexity_list_iter), "-r" ,label = 'train perplexity')
            plt.legend(loc="upper right")
            plt.grid()
            plt.show()

        else:
            if valid:
                plt.plot(to_cpu(self.valid_perplexity_list), "-b" ,label = 'valid perplexity')
                plt.title("train/valid perplexity per epoch")
            else:
                plt.title("train perplexity per epoch")

            plt.plot(to_cpu(self.train_perplexity_list), "-r" ,label = 'train perplexity')

            plt.legend(loc="upper right")

            plt.grid()
            plt.show()



    def add_metric(self, **metric):
        print(f"metic {metric.keys()} added")
        self.metric = dict(self.metric, metric)