import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from neuralflow.gpu import *
from neuralflow.utils import *
import time


class BaseTrainer():
    def __init__(self, n_epochs= 10, init_lr=0.01, file_name = None):
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
        self.train_loss_list_iter = np.array([])
        self.valid_loss_list = np.array([])
        self.valid_loss_list_iter = np.array([])
        self.test_loss_list = np.array([])
        self.test_loss_list_iter = np.array([])
        self.best_valid_loss = 100
        self.file_name = file_name
        self.train_time = np.array([])

    def __repr__(self):
        return "Trainer"

    
    def get_train_loss_list(self):
        return self.train_loss_list
    
    
    def get_train_loss_list_iter(self):
        return self.train_loss_list_iter
    
    
    def get_valid_loss_list(self):
        return self.valid_loss_list
    
    
    def get_train_loss_list_iter(self):
        return self.valid_loss_list_iter
    

class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, critic, optimizer, n_epochs= 10, init_lr=0.01, file_name= None):
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
        super().__init__(n_epochs, init_lr, file_name=file_name)
        self.model = model
        self.critic = critic
        self.optimizer = optimizer

        self.train_accuracy_list = np.array([0])
        self.valid_accuracy_list = np.array([0])
        self.test_accuracy_list = np.array([0])
        

    def _forward(self, x, y):
        pred = self.model(x)
        loss = self.critic(pred, y)
            
        return loss

    
    def _backward(self):
        self.model.backward(self.critic)
        
    
    def _update(self):
        self.optimizer.update(self.model)
    

    def train(self, train_dataloader, valid_dataloader = None, show_iter_num=1):
        """
        Train model with train/valid data

        Parameters
        ----------
        train_dataloader (iterable) : Train dataloader that can iterate with batch size in train dataset

        valid_dataloader (iterable) : Valid dataloader that can iterate with batch size in valid dataset

        """
        
        for epoch in range(self.n_epochs):
            start_time = time.time()
            print(f"epoch {epoch+1}")
            self.model.train_state()
            tmp_train_loss = np.array([])
            iter_tmp_train_loss = np.array([])
            count = 0
            iter_num = len(train_dataloader)
            train_correct_num = 0
            y_num = 0
            for x, y in tqdm(train_dataloader):
                count+=1
                train_loss = self._forward(x, y)
                self._backward()
                self._update()

                tmp_train_loss = np.append(tmp_train_loss, train_loss)
                iter_tmp_train_loss = np.append(iter_tmp_train_loss, train_loss)
                
                if count % show_iter_num == 0:
                    iter_train_loss = np.sum(iter_tmp_train_loss) / show_iter_num
                    iter_tmp_train_loss = np.array([])
                    self.train_loss_list_iter = np.append(self.train_loss_list_iter, iter_train_loss)                
                
                pred = np.argmax(self.critic.pred, axis=1)
                
                if y.ndim != 1:
                    y = np.argmax(y, axis=1)
                    
                train_correct_num += np.sum(pred==y)
                y_num += len(y)
                train_temp_accuracy = train_correct_num/y_num*100
                
                print(f"train loss : {train_loss:.6f}    train accuarcy : {train_temp_accuracy:.6f}    iter : {count}/{iter_num}\r", end="")
                

            epoch_train_loss = np.sum(tmp_train_loss) / len(tmp_train_loss)
            self.train_loss_list = np.append(self.train_loss_list, epoch_train_loss)

            dataset_len = train_dataloader.dataset_len()
            epoch_train_accuracy = train_correct_num/float(dataset_len) * 100
            self.train_accuracy_list = np.append(self.train_accuracy_list, epoch_train_accuracy)
            
            print()
            print(f"epoch {epoch+1} -- train loss : {epoch_train_loss}    train accuarcy : {epoch_train_accuracy}")
            
            finish_time = time.time()
            epoch_time = finish_time - start_time
            self.train_time = np.append(self.train_time, epoch_time)
            print(f"{epoch_time:.1f}s elapsed")
            if valid_dataloader is not None:
                self._validate(valid_dataloader, epoch, show_iter_num=show_iter_num)
            else:
                print("----------------------------------------------------------------")    


    def _validate(self, valid_dataloader, epoch, show_iter_num = 1):
        self.model.eval_state()
        self.model.reset_rnn_state()
        tmp_valid_loss = np.array([])
        iter_tmp_valid_loss = np.array([])
        count = 0
        valid_correct_num = 0
        y_num = 0
        for x, y in tqdm(valid_dataloader):
            count+=1
            valid_loss = self._forward(x, y)
            tmp_valid_loss = np.append(tmp_valid_loss, valid_loss)
            pred = np.argmax(self.critic.pred, axis=1)
            if y.ndim != 1:
                y = np.argmax(y, axis=1)
            valid_correct_num += np.sum(pred==y)
            y_num += len(y)
            
            valid_temp_accuracy = valid_correct_num/y_num*100
            iter_tmp_valid_loss = np.append(iter_tmp_valid_loss, valid_loss)
            
            if count % show_iter_num == 0:
                iter_valid_loss = np.sum(iter_tmp_valid_loss) / show_iter_num
                iter_tmp_valid_loss = np.array([])
                self.valid_loss_list_iter = np.append(self.valid_loss_list_iter, iter_valid_loss)

            print(f"valid loss : {valid_loss:.6f}    valid accuarcy : {valid_temp_accuracy:.6f}\r", end="")

        dataset_len = valid_dataloader.dataset_len()
        
        epoch_valid_loss = np.sum(tmp_valid_loss) / len(tmp_valid_loss)
        self.valid_loss_list = np.append(self.valid_loss_list, epoch_valid_loss)
        
        epoch_valid_accuracy = valid_correct_num/float(dataset_len) * 100
        self.valid_accuracy_list = np.append(self.valid_accuracy_list, epoch_valid_accuracy)
        self.model.reset_rnn_state()
        
        if epoch_valid_loss < self.best_valid_loss and self.file_name != None:
            self.model.save_params(self.file_name)
            self.best_valid_loss = epoch_valid_loss
        
        
        print()
        print(f"epoch {epoch+1} -- valid loss : {epoch_valid_loss}    valid accuarcy : {epoch_valid_accuracy}")
        print("----------------------------------------------------------------")    

        
    def show_error_graph(self, show_iter = False, valid = False):
        """
        Visualize training/validation error
        """
        if show_iter:
            
            if valid:
                plt.plot(to_cpu(self.valid_loss_list_iter), "-b" ,label = 'valid loss')
                plt.title("train/valid loss per iteration")
            else:
                plt.title("train loss per iteration")

            plt.plot(to_cpu(self.train_loss_list_iter), "-r" ,label = 'train loss')
            plt.legend(loc="upper right")
            plt.grid()
            plt.show()

        else:
            if valid:
                plt.plot(to_cpu(self.valid_loss_list), "-b" ,label = 'valid loss')
                plt.title("train/valid loss per epoch")
            else:
                plt.title("train loss per epoch")

            plt.plot(to_cpu(self.train_loss_list), "-r" ,label = 'train loss')

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
        
    
    def eval_accuracy(self, test_dataloader, show_iter_num = 1):
        self.model.eval_state()
        self.model.reset_rnn_state()
        tmp_test_loss = np.array([])
        iter_tmp_test_loss = np.array([])
        count = 0
        test_correct_num = 0
        y_num = 0
        for x, y in tqdm(test_dataloader):
            count+=1
            test_loss = self._forward(x, y)
            tmp_test_loss = np.append(tmp_test_loss, test_loss)
            pred = np.argmax(self.critic.pred, axis=1)
            if y.ndim != 1:
                y = np.argmax(y, axis=1)
            test_correct_num += np.sum(pred==y)
            y_num += len(y)
            
            iter_tmp_test_loss = np.append(iter_tmp_test_loss, test_loss)
            
            if count % show_iter_num == 0:
                iter_test_loss = np.sum(iter_tmp_test_loss) / show_iter_num
                iter_tmp_test_loss = np.array([])
                self.test_loss_list_iter = np.append(self.test_loss_list_iter, iter_test_loss)

        dataset_len = test_dataloader.dataset_len()
        
        epoch_test_loss = np.sum(tmp_test_loss) / len(tmp_test_loss)
        self.test_loss_list = np.append(self.test_loss_list, epoch_test_loss)
        
        epoch_test_accuracy = test_correct_num/float(dataset_len) * 100
        self.test_accuracy_list = np.append(self.test_accuracy_list, epoch_test_accuracy)
        self.model.reset_rnn_state()
        
        print()
        print(f"test loss : {epoch_test_loss}    test accuarcy : {epoch_test_accuracy}")
        print("----------------------------------------------------------------")
        
        
    def get_train_accuracy_list(self):
        return self.train_accuracy_list
    
    
    def get_valid_accuracy_list(self):
        return self.valid_accuracy_list
    
        
class LanguageModelTrainer(BaseTrainer):
    def __init__(self, model, critic, optimizer, n_epochs= 10, init_lr=0.01, file_name= None):
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
        super().__init__(n_epochs, init_lr, file_name=file_name)
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
    

    def train(self, train_dataloader, valid_dataloader = None, max_grad = None, show_iter_num = 1):
        """
        Train model with train/valid data

        Parameters
        ----------
        train_dataloader (iterable) : Train dataloader that can iterate with batch size in train dataset

        valid_dataloader (iterable) : Valid dataloader that can iterate with batch size in valid dataset

        """
        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train_state()
            print(f"epoch {epoch+1}")
            tmp_train_loss = np.array([])
            iter_tmp_train_loss = np.array([])
            tmp_train_perplexity = np.array([])
            iter_tmp_train_perplexity = np.array([])

            count = 0
            iter_num = len(train_dataloader)
            
            for x, y in tqdm(train_dataloader):
                count +=1
                # train loss
                train_loss = self._forward(x, y)
                self._backward()
                if max_grad is not None:
                    self.clip_grad(max_grad)
                self._update()
                tmp_train_loss = np.append(tmp_train_loss, train_loss)
                # train perplexity
                train_perplexity = np.exp(train_loss)
                tmp_train_perplexity = np.append(tmp_train_perplexity, train_perplexity)
                iter_tmp_train_perplexity = np.append(iter_tmp_train_perplexity, train_perplexity)
                iter_tmp_train_loss = np.append(iter_tmp_train_loss, train_loss)
                
                if count % show_iter_num == 0:
                    iter_train_loss = np.sum(iter_tmp_train_loss) / show_iter_num
                    iter_tmp_train_loss = np.array([])
                    self.train_loss_list_iter = np.append(self.train_loss_list_iter, iter_train_loss)
                    
                    iter_train_perplexity = np.sum(iter_tmp_train_perplexity) / show_iter_num
                    iter_tmp_train_perplexity = np.array([])
                    self.train_perplexity_list_iter = np.append(self.train_perplexity_list_iter, iter_train_perplexity)

                print(f"train loss : {train_loss:.6f}    train perplexity : {train_perplexity:.6f}    iter : {count}/{iter_num}\r", end="")
                
            
            epoch_train_loss = np.sum(tmp_train_loss) / count
            self.train_loss_list = np.append(self.train_loss_list, epoch_train_loss)
            
            
            epoch_train_perplexity = np.sum(tmp_train_perplexity) / count
            self.train_perplexity_list = np.append(self.train_perplexity_list, epoch_train_perplexity)
            
            print()
            print(f"epoch {epoch+1} -- train loss : {epoch_train_loss:.6f}    train perplexity : {epoch_train_perplexity:.6f}")
            finish_time = time.time()
            epoch_time = finish_time - start_time
            self.train_time = np.append(self.train_time, epoch_time)
            print(f"{epoch_time:.1f}s elapsed")
            
            if valid_dataloader is not None:
                self._validate(valid_dataloader, epoch, show_iter_num=show_iter_num)
            else:
                print("----------------------------------------------------------------")
                print()


    def _validate(self, valid_dataloader, epoch, show_iter_num = 1):
        self.model.eval_state()
        self.model.reset_rnn_state()
        tmp_valid_loss = np.array([])
        iter_tmp_valid_loss = np.array([])
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
            
            if count % show_iter_num == 0:
                iter_valid_loss = np.sum(iter_tmp_valid_loss) / show_iter_num
                iter_tmp_valid_loss = np.array([])
                self.valid_loss_list_iter = np.append(self.valid_loss_list_iter, iter_valid_loss)
                
                iter_valid_perplexity = np.sum(iter_tmp_valid_perplexity) / show_iter_num
                iter_tmp_valid_perplexity = np.array([])
                self.valid_perplexity_list_iter = np.append(self.valid_perplexity_list_iter, iter_valid_perplexity)
            
            print(f"valid loss : {valid_loss:.6f}    valid perplexity : {valid_perplexity:.6f}\r", end="")
            
        epoch_valid_loss = np.sum(tmp_valid_loss) / count
        self.valid_loss_list = np.append(self.valid_loss_list, epoch_valid_loss)
        
        epoch_valid_perplexity = np.sum(tmp_valid_perplexity) / count
        self.valid_perplexity_list = np.append(self.valid_perplexity_list, epoch_valid_perplexity)
        self.model.reset_rnn_state()

        if epoch_valid_loss < self.best_valid_loss and self.file_name != None:
            self.model.save_params(self.file_name)
            self.best_valid_loss = epoch_valid_loss        

        print()
        print(f"epoch {epoch+1} -- valid loss : {epoch_valid_loss:.6f}    valid perplexity : {epoch_valid_perplexity:.6f}")
        print("----------------------------------------------------------------")
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
                


    def show_error_graph(self, show_iter = False, valid = False):
        """
        Visualize training/validation error
        """
        if show_iter:
            
            if valid:
                plt.plot(to_cpu(self.valid_loss_list_iter), "-b" ,label = 'valid loss')
                plt.title("train/valid loss per iteration")
            else:
                plt.title("train loss per iteration")

            plt.plot(to_cpu(self.train_loss_list_iter), "-r" ,label = 'train loss')
            plt.legend(loc="upper right")
            plt.grid()
            plt.show()

        else:
            if valid:
                plt.plot(to_cpu(self.valid_loss_list), "-b" ,label = 'valid loss')
                plt.title("train/valid loss per epoch")
            else:
                plt.title("train loss per epoch")

            plt.plot(to_cpu(self.train_loss_list), "-r" ,label = 'train loss')

            plt.legend(loc="upper right")

            plt.grid()
            plt.show()


    def show_perplexity_graph(self, show_iter = False, valid = False):
        """
        Visualize training/validation perplexity
        """
        plt.ylim(-1, 401)
        if show_iter:
            
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


    def eval_perplexity(self, test_dataloader):
        print('Test perplexity ...')
        self.model.eval_state()
        self.model.reset_rnn_state()
        
        tmp_test_loss = np.array([])
        tmp_test_perplexity = np.array([])

        count = 0
        for x, y in tqdm(test_dataloader):
            count +=1
            # test loss
            test_loss = self._forward(x, y)
            tmp_test_loss = np.append(tmp_test_loss, test_loss)
            # test perplexity
            test_perplexity = np.exp(test_loss)
            tmp_test_perplexity = np.append(tmp_test_perplexity, test_perplexity)
            
        epoch_test_loss = np.sum(tmp_test_loss) / count
        epoch_test_perplexity = np.sum(tmp_test_perplexity) / count

        print()
        print(f"test loss : {epoch_test_loss:.6f}    test perplexity : {epoch_test_perplexity:.6f}")
        
        
    def get_train_ppl_list(self):
        return self.train_perplexity_list
    
    
    def get_train_ppl_list_iter(self):
        return self.train_perplexity_list_iter
    
    
    def get_valid_ppl_list(self):
        return self.valid_perplexity_list
    
    
    def get_train_ppl_list_iter(self):
        return self.valid_perplexity_list_iter
    
    
class Seq2SeqTrainer(BaseTrainer):
    def __init__(self, model, critic, optimizer, vocab, n_epochs= 10, init_lr=0.01, reverse=False, file_name= None):
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
        super().__init__(n_epochs, init_lr, file_name=file_name)
        self.model = model
        self.critic = critic
        self.optimizer = optimizer
        self.vocab = vocab
        self.verbose = 0
        self.is_reverse = reverse
        
        self.train_accuracy_list = np.array([0])
        self.valid_accuracy_list = np.array([0])
        self.test_accuracy_list = np.array([0])
        

    def _forward(self, x, y):
        decoder_x, decoder_y = y[:, :-1], y[:, 1:]
        pred = self.model(x, decoder_x)
        loss = self.critic(pred, decoder_y)
        
        return loss

    
    def _backward(self):
        self.model.backward(self.critic)
        
    
    def _update(self):
        self.optimizer.update(self.model)
    

    def train(self, train_dataloader, valid_x, valid_y, max_grad = None, show_iter_num=1, verbose = 10):
        """
        Train model with train/valid data

        Parameters
        ----------
        train_dataloader (iterable) : Train dataloader that can iterate with batch size in train dataset

        valid_dataloader (iterable) : Valid dataloader that can iterate with batch size in valid dataset

        """
        for epoch in range(self.n_epochs):
            self.verbose = verbose
            start_time = time.time()
            print(f"epoch {epoch+1}")
            self.model.train_state()
            
            tmp_train_loss = np.array([])
            iter_tmp_train_loss = np.array([])
            count = 0
            iter_num = len(train_dataloader)
            train_correct_num = 0
            y_num = 0
            
            for x, y in tqdm(train_dataloader):
                count+=1
                train_loss = self._forward(x, y)
                self._backward()
                if max_grad is not None:
                    self.clip_grad(max_grad)
                self._update()

                tmp_train_loss = np.append(tmp_train_loss, train_loss)
                iter_tmp_train_loss = np.append(iter_tmp_train_loss, train_loss)
         
                print(f"train loss : {train_loss:.6f}    iter : {count}/{iter_num}\r", end="")

            epoch_train_loss = np.sum(tmp_train_loss) / len(tmp_train_loss)
            self.train_loss_list = np.append(self.train_loss_list, epoch_train_loss)

            print()
            print(f"epoch {epoch+1} -- train loss : {epoch_train_loss}")
            
            finish_time = time.time()
            epoch_time = finish_time - start_time
            self.train_time = np.append(self.train_time, epoch_time)
            print(f"{epoch_time:.1f}s elapsed")
            
            self.model.eval_state()
            correct_num = 0
            for i in range(len(valid_x)):
                question, correct = valid_x[[i]], valid_y[[i]]
                correct_num += self.eval_seq2seq(question, correct)
                
                print(f"valid accuarcy : {correct_num/len(valid_x):.6f}\r", end="")
        
            epoch_valid_accuracy = (correct_num/float(len(valid_x))) * 100
            self.valid_accuracy_list = np.append(self.valid_accuracy_list, epoch_valid_accuracy)
        
            print()
            print(f"epoch {epoch+1} -- valid accuarcy : {epoch_valid_accuracy}")
            print("----------------------------------------------------------------")  
            


    # def _validate(self, valid_dataloader, epoch, show_iter_num = 1):
    #     self.model.eval_state()
    #     valid_correct_num = 0
    #     dataset_len = valid_dataloader.dataset_len()
    #     for x, y in tqdm(valid_dataloader):
    #         valid_correct_num += self.eval_seq2seq(x, y)
            
    #         valid_temp_accuracy = valid_correct_num/dataset_len*100
            
    #         print(f"valid accuarcy : {valid_temp_accuracy:.6f}\r", end="")
        
    #     epoch_valid_accuracy = valid_correct_num/float(dataset_len) * 100
    #     self.valid_accuracy_list = np.append(self.valid_accuracy_list, epoch_valid_accuracy)
        
    #     print()
    #     print(f"epoch {epoch+1} -- valid accuarcy : {epoch_valid_accuracy}")
    #     print("----------------------------------------------------------------")    

    

    
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
                        
        
    def eval_seq2seq(self, src, tgt):
        tgt = tgt.flatten()
        start_id = tgt[0]
        tgt = tgt[1:]
        pred = self.model.generate(src, start_id, len(tgt))
        
  
        src = ''.join([self.vocab[int(c)] for c in src.flatten()])
        tgt = ''.join([self.vocab[int(c)] for c in tgt])
        pred = ''.join([self.vocab[int(c)] for c in pred])

        if self.verbose > 0:
            if self.is_reverse:
                src = src[::-1]

            colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
            print('Q', src)
            print('T', tgt)

            is_windows = os.name == 'nt'

            if pred == tgt:
                mark = colors['ok'] + '☑' + colors['close']
                if is_windows:
                    mark = 'O'
                print(mark + ' ' + pred)
            else:
                mark = colors['fail'] + '☒' + colors['close']
                if is_windows:
                    mark = 'X'
                print(mark + ' ' + pred)
            print('---')
            
            self.verbose -= 1
                    
        return 1 if pred == tgt else 0
        
        
    def get_train_accuracy_list(self):
        return self.train_accuracy_list
    
    
    def get_valid_accuracy_list(self):
        return self.valid_accuracy_list