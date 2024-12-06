import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader ,TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch import optim
# from models.LSTMAE import LSTMAE
from TransAE import TransformerAE
#from TransAE2 import WaveletTransformerAE 
from train_utils import train_model, eval_model,test_model
from Unit.utils import get_from_one ,metrics_calculate,anomaly_scoring,evaluate,get_from_all
# from ranger import Ranger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping2 import EarlyStopping
# from WTConv import WTConv1d 
early_stopping = EarlyStopping('./earlysave')

parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate')#0.001
parser.add_argument('--input-size', type=int, default=25, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=1, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='TransAE', help='currently only TransAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=50, help='sequence full size')
parser.add_argument('--datapath',default='./data/PSM',help='datapath')
parser.add_argument('--dataset',default="PSM",help='data')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args = parser.parse_args(args=[])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# folder settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)


class toy_dataset(torch.utils.data.Dataset):
    def __init__(self, toy_data):
        self.toy_data = toy_data

    def __len__(self):
        return self.toy_data.shape[0]

    def __getitem__(self, index):
        return self.toy_data[index]


def main():

    # Create data loaders
    early = 'trans'

    states = 0
    train_iter, val_iter = create_dataloaders(args.batch_size)
######PSM#########
    input_dim = 25   
    embed_dim = 64  
    num_heads = 4    
    num_layers = 1   
    ff_dim = 128 
        ######SMAP#########
    # input_dim = 25   
    # embed_dim = 64  
    # num_heads = 4   
    # num_layers = 2  
    # ff_dim = 128 
        ######SWAT########
    # input_dim = 51   # 输入特征维度
    # embed_dim = 64  # 嵌入维度
    # num_heads = 4    # 多头注意力的头数
    # num_layers = 1   # 编码器/解码器的层数
    # ff_dim = 128  
        ######WADI########
    # input_dim = 127   # 输入特征维度
    # embed_dim = 128  # 嵌入维度
    # num_heads = 8    # 多头注意力的头数
    # num_layers = 1   # 编码器/解码器的层数
    # ff_dim = 256  

    # Create model
    #model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout, seq_len=args.seq_len)
    #model = WaveletTransformerAE(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)

    model = TransformerAE(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
                      num_layers=num_layers, ff_dim=ff_dim)
    
    model.to(device)
    #wt = WTConv1d(in_channels= input_dim, out_channels= input_dim).to(device)

    # 打印模型的参数数目
    params = list(model.parameters())
    print(f"Total number of parameters: {len(params)}")  # 检查参数数目
    if len(params) == 0:
       print("Model has no parameters registered.")


    # Create optimizer & loss functions
    optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = Ranger(params=model.parameters(), lr=args.lr, weight_decay=args.wd )
    # 设定学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    criterion = torch.nn.MSELoss(reduction='sum')


    # Grid search run if run-grid-search flag is active
    if args.run_grid_search:
        hyper_params_grid_search(train_iter, val_iter, criterion)
        return

    # Train & Val
    for epoch in range(args.epochs):
        # Train loop
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss, train_acc, train_pred_loss = train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, args.grad_clipping,
                    args.log_interval)

        val_loss, val_acc = eval_model(criterion, model,args.model_type, val_iter)
        scheduler.step(val_loss)
        early_stopping(val_loss, model,states,early)

        if early_stopping.early_stop :
            print("*******************EMB early stop*********************")
            if (args.dataset == 'SMAP'):
                print('load SMAP')
                testdata = np.load('./data/SMAP/SMAP/SMAP_test.npy')
                testdata = testdata.astype(np.float32)
                # scaler = MinMaxScaler(feature_range=(0, 1))  # 定义MinMaxScaler并设置范围为0-1
                # testdata = scaler.fit_transform(testdata)    
                label = np.load('./data/SMAP/SMAP/SMAP_test_label.npy')
                # testdata = testdata[:135000]
                # label = label[:135000]



            elif (args.dataset == 'WADI'):
                testdata = pd.read_csv('./data/WADI/WADI_test.csv')
                testdata = testdata.values[:, 1:-1]
                testdata = testdata.astype(np.float32)
                testdata = np.nan_to_num(testdata)
                scaler = StandardScaler()
                testdata = scaler.fit_transform(testdata)
                
                label = pd.read_csv('./data/WADI/WADI_label.csv')
                label = label.values[:, :]


            elif (args.dataset == 'SWAT'):

                testdata = pd.read_csv('./data/SWAT/SWaT_test.csv')
                testdata =testdata.fillna(testdata.mean())
                testdata = testdata.values[:, 0:-2]
                testdata = testdata.astype(np.float32)
                scaler = MinMaxScaler(feature_range=(0, 1))  # 定义MinMaxScaler并设置范围为0-1
                testdata = scaler.fit_transform(testdata)
                label = pd.read_csv('./data/SWAT/SWaT_test.csv')
                label=label.values[:,-1]


            elif (args.dataset == 'PSM'):
                testdata = pd.read_csv('./data/PSM/PSM/test.csv')
                testdata =testdata.fillna(testdata.mean())
                testdata = testdata.values[:, 1:]
                testdata=testdata.astype(np.float32)
                scaler = MinMaxScaler(feature_range=(0, 1))  # 定义MinMaxScaler并设置范围为0-1
                testdata = scaler.fit_transform(testdata)
                label = pd.read_csv('./data/PSM/PSM/test_label.csv')
                label = label.values[:, 1:]


            label = label.astype(None)
            label = torch.tensor(label).to(device)

            # Make sure model is on the correct device
            model = model.to(device)
            model.eval()

            
            real_data = torch.Tensor(testdata).to(device)  # Move real_data to the correct device
            #testdata = get_from_one(testdata, window_size=64, stride=1)

            # Create DataLoader
            dataloader = DataLoader(
                testdata, batch_size=128, shuffle=True, num_workers=16, drop_last=True,
                pin_memory=True)
            

            test_model(model,dataloader,real_data,label)

            # with torch.no_grad():
            #     for data in dataloader:
            #         if len(data) == 2:
            #             data, labels = data[0].to(device), data[1].to(device)
            #         else:
            #             data = data.to(device)
                    
            #         data = data.type(torch.FloatTensor).to(device)  # Ensure data is on the correct device
            #         model_out = model(data, 'all')
            #         #re_datas.extend(data)
            #         re_datas.extend(model_out.cpu().detach().numpy())
            #         #re_datas.extend(model_out.cpu().detach().numpy())  # Move outputs to CPU if necessary

            # # Convert re_datas back to tensor and move to the same device
            # #re_datas = torch.tensor(np.array([item.cpu().detach().numpy() for item in re_datas]))
            # re_datas = torch.tensor(np.array(re_datas))
            # #re_datas_cpu = re_datas.cpu().detach().numpy()  # Move to CPU and convert to NumPy
            # real_data = real_data.cpu().detach().numpy()  # Move to CPU and convert to NumPy
            # label = label.cpu().detach().numpy() 
            # # Make sure label and real_data are the same length as re_datas
            # label = label[:len(re_datas)]
            # real_data = real_data[:len(re_datas)]

            # # Ensure all inputs for the metrics calculation are on the same device
            # metrics_calculate(real_data, re_datas, label, mind=None)


            break
   
    print("END")



def create_toy_data(num_of_sequences=10000, sequence_len=64) -> torch.tensor:
    """
    Generate num_of_sequences random sequences with length of sequence_len each.
    :param num_of_sequences: number of sequences to generate
    :param sequence_len: length of each sequence
    :return: pytorch tensor containing the sequences
    """
    
    path = args.datapath

    if (args.dataset == 'SMAP'):
            print('Load SMAP')
            toy_data = np.load('./data/SMAP/SMAP/SMAP_train.npy')
            toy_data=toy_data.astype(np.float32)
            # scaler = MinMaxScaler(feature_range=(0, 1))  # 定义MinMaxScaler并设置范围为0-1
            # toy_data = scaler.fit_transform(toy_data)
            ws = 64


    elif (args.dataset == 'WADI'):
            print('Load WADI')
            toy_data = pd.read_csv('./data/WADI/WADI_train.csv')
            toy_data=toy_data.values[:, 1:-1]
            toy_data=toy_data.astype(np.float32)
            toy_data = np.nan_to_num(toy_data)
            scaler = StandardScaler()
            toy_data = scaler.fit_transform(toy_data)

            ws = 64


    elif (args.dataset == 'SWAT'):
            print('Load SWAT')
            toy_data = pd.read_csv('./data/SWAT/SWaT_train.csv')
            toy_data = toy_data.fillna(toy_data.mean())
            toy_data=toy_data.values[:, 0:-1]
            toy_data=toy_data.astype(np.float32)
            
            scaler = MinMaxScaler(feature_range=(0, 1))  # 定义MinMaxScaler并设置范围为0-1
            toy_data = scaler.fit_transform(toy_data)
            ws= 64


    elif (args.dataset == 'PSM'):
            
            print('Load PSM')
            toy_data = pd.read_csv('./data/PSM/PSM/train.csv')
            toy_data = toy_data.fillna(toy_data.mean())
            toy_data = toy_data.values[:, 1:]
            toy_data=toy_data.astype(np.float32)
            # toy_data = np.nan_to_num(toy_data)
            # scaler = StandardScaler()
            # toy_data = scaler.fit_transform(toy_data)
            scaler = MinMaxScaler(feature_range=(0, 1))  # 定义MinMaxScaler并设置范围为0-1
            toy_data = scaler.fit_transform(toy_data)
            ws = 64

    # print('Load WADI')
    # toy_data = np.load('./data/WADI/WADI_train.npy',allow_pickle=True)
    # toy_data = np.nan_to_num(toy_data)
    # scaler = StandardScaler()
    # toy_data = scaler.fit_transform(toy_data)
    # length = int(len(toy_data))
    # toy_data = toy_data[:length]
    # print(toy_data.shape)
    toy_data = get_from_one(toy_data,window_size=ws,stride=1)
    # print(toy_data.shape)
    toy_data = torch.tensor(toy_data).float()


    return toy_data


def create_dataloaders(batch_size, train_ratio=0.9):
    """
    Build train, validation and tests dataloader using the toy data
    :return: Train, validation and test data loaders
    
    """
    val_ratio=1-train_ratio
    toy_data = create_toy_data()
    print('************')
    print(toy_data.shape)
    len = toy_data.shape[0]

    train_data = toy_data[:int(len * train_ratio), :]
    val_data = toy_data[int(train_ratio * len):int(len * (train_ratio + val_ratio)), :]
   

    print(f'Datasets shapes: Train={train_data.shape}; Validation={val_data.shape}')
    train_iter = torch.utils.data.DataLoader(toy_dataset(train_data), batch_size=batch_size, drop_last=True,shuffle=True, num_workers=8)
    val_iter = torch.utils.data.DataLoader(toy_dataset(val_data), batch_size=batch_size,  drop_last=True,shuffle=True, num_workers=8)
  

    return train_iter, val_iter


def plot_toy_data(toy_example, description, color='b'):
    """
    Recieves a toy raw data sequence and plot it
    :param toy_example: toy data example sequence
    :param description: additional description to the plot
    :param color: graph color
    :return:
    """
    time_lst = [t for t in range(toy_example.shape[0])]

    plt.figure()
    plt.plot(time_lst, toy_example.tolist(), color=color)
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    # plt.legend()
    plt.title(f'Single value vs. time for toy example {description}')
    plt.show()


def plot_orig_vs_reconstructed(model, test_iter,modelpath, num_to_plot=2):
    """
    Plot the reconstructed vs. Original MNIST figures
    :param model: model trained to reconstruct MNIST figures
    :param test_iter: test data loader
    :param num_to_plot: number of random plots to present
    :return:
    """



    # Plot original and reconstructed toy data
    plot_test_iter = iter(torch.utils.data.DataLoader(test_iter.dataset, batch_size=1, shuffle=False))

    for i in range(num_to_plot):
        orig = next(plot_test_iter).to(device)
        with torch.no_grad():
            rec = model(orig,'all')
            print(rec.shape)

        time_lst = [t for t in range(orig.shape[1])]

        # Plot original
        plot_toy_data(orig.squeeze(), f'Original sequence #{i + 1}', color='g')

        # Plot reconstruction
        plot_toy_data(rec.squeeze(), f'Reconstructed sequence #{i + 1}', color='r')

        # Plot combined
        plt.figure()
        plt.plot(time_lst, orig.squeeze().tolist(), color='g', label='Original signal')
        plt.plot(time_lst, rec.squeeze().tolist(), color='r', label='Reconstructed signal')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.legend()
        title = f'Original and Reconstruction of Single values vs. time for toy example #{i + 1}'
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.show()


def hyper_params_grid_search(train_iter, val_iter, criterion):
    """
    Function to perform hyper-parameter grid search on a pre-defined range of values.
    :param train_iter: train dataloader
    :param val_iter: validation data loader
    :param criterion: loss criterion to use (MSE for reconstruction)
    :return:
    """
    lr_lst = [1e-2, 1e-3, 1e-4]
    hs_lst = [16, 32, 64, 128, 256]
    clip_lst = [None, 10, 1]

    total_comb = len(lr_lst) * len(hs_lst) * len(clip_lst)
    print(f'Total number of combinations: {total_comb}')

    curr_iter = 1
    best_param = {'lr': None, 'hs': None, 'clip_val': None}
    best_val_loss = np.Inf
    params_loss_dict = {}

    for lr in lr_lst:
        for hs in hs_lst:
            for clip_val in clip_lst:
                print(f'Starting Iteration #{curr_iter}/{total_comb}')
                curr_iter += 1
                model = LSTMAE(input_size=args.input_size, hidden_size=hs, dropout_ratio=args.dropout,
                               seq_len=args.seq_len)
                model = model.to(device)
                optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=lr, weight_decay=args.wd)

                for epoch in range(args.epochs):
                    # Train loop
                    train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, clip_val,
                                args.log_interval)
                avg_val_loss, val_acc = eval_model(criterion, model, args.model_type, val_iter)
                params_loss_dict.update({f'lr={lr}_hs={hs}_clip={clip_val}': avg_val_loss})
                if avg_val_loss < best_val_loss:
                    print(f'Found better validation loss: old={best_val_loss}, new={avg_val_loss}; parameters: lr={lr},hs={hs},clip_val={clip_val}')
                    best_val_loss = avg_val_loss
                    best_param = {'lr': lr, 'hs': hs, 'clip_val': clip_val}

    print(f'Best parameters found: {best_param}')
    print(f'Best Validation Loss: {best_val_loss}')
    print(f'Parameters loss: {params_loss_dict}')


if __name__ == '__main__':
    main()
