import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience= 1, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss,model,states,early):
        
        print('early stop change')
        
        if early=='trans':
            score = val_loss/100
            if self.best_score is None:
                self.best_score = score
                self.savemodel(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print('************************************************************')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.savemodel(val_loss, model)
                self.counter = 0

            print('best_score={}'.format(self.best_score))

        elif early=='unet':
            score = val_loss
            if self.best_score is None:
                self.best_score = score
                self.savemodel2(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print('************************************************************')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.savemodel2(val_loss, model)
                self.counter = 0

            print('best_score={}'.format(self.best_score))


        elif early =='ddim':
            score = val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, states)
            elif score > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print('************************************************************')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, states)
                self.counter = 0

            print('best_score={}'.format(self.best_score))



    #save the diffusion model
    def save_checkpoint(self, val_loss,states):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_newPSM_DMnetwork.pth')#best_newPSM_DMnetwork.pth
        torch.save(states, path)

        self.val_loss_min = val_loss

    #save the trans model
    def savemodel(self,val_loss,model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, 'best_newPSM_Transnetwork.pth')#best_newPSM_network.pth
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
    
    #save the unet model
    def savemodel2(self,val_loss,model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, 'best_newSMAP_Unetwork.pth')#best_newPSM_Unetwork.pth
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
