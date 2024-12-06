import os
import logging
import time
import glob
import sys
from matplotlib import pyplot as plt
from Unit.utils import get_from_one,metrics_calculate
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from models.LSTMAE import LSTMAE
from TransAE import TransformerAE
from models.Unet import UNet
from models.diffusion import Model,Model2
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry,noise_estimation_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from ranger import Ranger
import torchvision.utils as tvu
import argparse
# from WTConv import WTConv1d 
from pytorch_wavelets import DWT1DForward, DWT1DInverse
# from SelfAtten import AttentionFusionModel


from early_stopping2 import EarlyStopping
# The terminal log output is disabled
logging.getLogger().handlers.clear()

# 或者添加一个空处理器（不会输出任何东西）
logging.getLogger().addHandler(logging.NullHandler())
logging.basicConfig(level=logging.WARNING)

logging.info("This will not be shown in the terminal.")
logging.warning("This will be shown in the terminal.")

early_stopping = EarlyStopping('./earlysave')   



parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default= 16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='AdamW', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')#0.001
parser.add_argument('--input-size', type=int, default=25, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=5, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=50, help='sequence full size')
parser.add_argument('--datapath',default='./data/SMAP/SMAP/SMAP_train.npy',help='datapath')
parser.add_argument('--dataset',default="PSM",help='data')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args2 = parser.parse_args(args=[])


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

# def get_from_all(samples, window_size, stride):
#     num_samples, window_size_sample, feature_size = samples.shape
#     ts_length = (num_samples - 1) * stride + window_size_sample 
#     ts_reconstructed = np.zeros((ts_length, feature_size), dtype=np.float64) 

#     for i, sample in enumerate(samples):
#         start = i * stride
#         sample = np.asarray(sample, dtype=np.float64)
#         ts_slice = ts_reconstructed[start:start + window_size_sample]
#         ts_slice = np.asarray(ts_slice, dtype=np.float64)
#         ts_reconstructed[start:start + window_size_sample] = ts_slice + sample
#         ts_counts[start:start + window_size_sample] += 1  
#     ts_counts = np.maximum(ts_counts, 1)
#     ts_counts = ts_counts[:, np.newaxis] 
#     ts_reconstructed /= ts_counts  
#     return ts_reconstructed






def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
           
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()


    def wavefussion(self):
        args, config = self.args, self.config

        #Load data
        if (args2.dataset == 'SMAP'):
            print('Load SMAP')
            dataset = np.load('./data/SMAP/SMAP/SMAP_train.npy')
            scaler = MinMaxScaler(feature_range=(0, 1))  
            dataset = scaler.fit_transform(dataset)

            ckpt1 = torch.load('./earlysave10/best_newSMAP_Transnetwork.pth',
                               map_location=self.device)
            length = int(dataset.shape[0]*0.9)
            traindata = dataset[:length]
            testdata = dataset[length:]


        elif (args2.dataset == 'WADI'):
            print('Load WADI')
            dataset = pd.read_csv('./data/WADI/WADI_train.csv')
            dataset = dataset.fillna(dataset.mean())
            dataset = dataset.fillna(0)
            dataset=dataset.values[:, 1:-1]
            dataset=dataset.astype(np.float32)
            scaler = MinMaxScaler(feature_range=(0, 1)) 
            dataset = scaler.fit_transform(dataset)

        elif (args2.dataset == 'SWAT'):
            print('Load SWAT')
            dataset = pd.read_csv('./data/SWAT/SWaT_train.csv')
            dataset = dataset.fillna(dataset.mean())
            dataset=dataset.values[:, 0:-1]
            dataset=dataset.astype(np.float32)
            scaler = MinMaxScaler(feature_range=(0, 1))  
            dataset = scaler.fit_transform(dataset)

            ckpt1 = torch.load(
                './earlysave10/best_newSWAT_Transnetwork.pth',
                map_location=self.device)
            length = int(dataset.shape[0]* 0.95)
            testdata = dataset[length:]
            traindata = dataset[:length]


            #
        elif (args2.dataset == 'PSM'):
            print('Load PSM')
            dataset = pd.read_csv('./data/PSM/PSM/train.csv')
            dataset =dataset.fillna(dataset.mean())
            dataset = dataset.values[:, 1:]
            dataset=dataset.astype(np.float32)
            scaler = MinMaxScaler(feature_range=(0, 1))  
            dataset = scaler.fit_transform(dataset)
            

            length = int(dataset.shape[0]*0.9)
            traindata = dataset[:length]
            testdata = dataset[length:]

            ckpt1 = torch.load(
                './earlysave/best_newPSM_Transnetwork.pth',
                map_location=self.device)

            label = pd.read_csv('./data/PSM/PSM/test_label.csv')
            label = label.values[:, 1:]
            label = label.astype(None)
            

        #  #Load unet

        # unet = UNet(config).to(self.device)
        # states = torch.load(
        #             './earlysave10/best_newPSM_Unetwork.pth', 
        #             map_location=self.config.device,
        #         )#./earlysave4/best_newPSM_Unetwork.pth-PSM
        # unet.load_state_dict(states)
        # unet.eval()

        #load the TransAE
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
        # input_dim = 51  
        # embed_dim = 64 
        # num_heads = 4  
        # num_layers = 1   
        # ff_dim = 128  
        ######WADI########
        # input_dim = 127  
        # embed_dim = 128  
        # num_heads = 8    
        # num_layers = 1  
        # ff_dim = 256  


        transz = TransformerAE(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
                      num_layers=num_layers, ff_dim=ff_dim)
        transz.to(self.device)
        transz.load_state_dict(ckpt1)




        # tb_logger = self.config.tb_logger

        #window data
        windowsize = 64
        stride = 1

        real_data = torch.Tensor(testdata)

        traindata = get_from_one(traindata, window_size=windowsize, stride=stride)
        #testdata = get_from_one(testdata, window_size=windowsize, stride=stride)


        train_loader = torch.utils.data.DataLoader(
            traindata,
            batch_size=config.training.batch_size,
            # batch_size=8,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            testdata,
            batch_size=config.training.batch_size,
            # batch_size=8,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )

        # Define the wavelet transform and inverse transformation
        dwt1d = DWT1DForward(J=1, mode='zero', wave='db1').to(self.device)
        idwt1d = DWT1DInverse(mode='zero', wave='db1').to(self.device)

        model_cd = Model2(config).to(self.device)
        # optimizer_cd = Ranger(params=model_cd.parameters(), lr=self.config.optim.lr, weight_decay=1e-4)
        optimizer_cd = torch.optim.AdamW(params=model_cd.parameters(), lr=self.config.optim.lr, weight_decay=1e-4)
    

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model_cd)
        else:
            ema_helper = None

        start_epoch, step = 1, 0
        # whether to load pre-trained model
        if self.args.resume_training:

            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model_cd.load_state_dict(states[0])
            optimizer_cd.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        #time
        datafirst = time.time()
    


        for epoch in range(start_epoch, self.config.training.n_epochs):
            earlyloss = 0
            # print("This is {} epoch\n".format(epoch))
            num=0
            loss_sum=0

            data_start = time.time()
            data_time = 0
            for i, x in enumerate(train_loader):
                x =  x.to(self.device)
                n = x.size(0)
                data_time += time.time() - data_start
                #model_ca.train()
                model_cd.train()
                step += 1
                num+=1
                x = x.type(torch.FloatTensor)
                x = x.to(self.device)
                #x = unet(x)#[128, 64, 25]
                z = transz(x,'en')#[128, 64, 16][B,W,C]
                z = z.permute(0, 2, 1) #[128, 16, 64][B,C,W]

                # z_sample_transformed = z[0]  # Take the first sample [64, 64]

                # # Visualize the z of shape [64, 64]
                # z_sample_transformed = z_sample_transformed.cpu().detach().numpy()  
                # plt.figure(figsize=(10, 6))
                # plt.imshow(z_sample_transformed, aspect='auto', cmap='plasma')
                # plt.colorbar(label='Latent Space Z values after Transformer')
                # plt.title("Visualization of Latent Space Z (Transformed Output)")
                # plt.xlabel("time (64)")
                # plt.ylabel("channel (64)")
                # #plt.show()
                # plt.savefig('z.png')
                
                coeffs = dwt1d(z)
                ca, cd_list = coeffs  # [128, 25, 32]
                # 可视化低频成分 (ca)
                # ca1 = ca[0].cpu().detach().numpy()
                # plt.figure(figsize=(10, 6))
                # plt.imshow(ca1, aspect='auto', cmap='plasma')
                # plt.colorbar(label='Low-frequency component (ca)')
                # plt.title("Low-frequency Component after Wavelet Transform")
                # plt.xlabel("time")
                # plt.ylabel("channel")
                # plt.savefig('ca.png')
                #print(ca.shape)

                cd = cd_list[0]  #  J=1，cd_list Only one element# [128, 25, 32]

                # cd1 = cd[0].cpu().detach().numpy()
                # plt.figure(figsize=(10, 6))
                # plt.imshow(cd1, aspect='auto', cmap='plasma')
                # plt.colorbar(label='High-frequency component (cd)')
                # plt.title("high-frequency Component after Wavelet Transform")
                # plt.xlabel("time ")
                # plt.ylabel("channel")
                # plt.savefig('cd.png')
                # plt.close('all')

                b = self.betas
                #e_ca = torch.randn_like(ca)
                e_cd = torch.randn_like(cd)
                
                # e_cd = torch.randn(cd.shape[0], cd.shape[1], cd.shape[2] * 2).to(self.device)
                # coeffs_noise = dwt1d(e_cd)
                # _, cd__list_noise = coeffs_noise
                # e_cd = cd__list_noise[0]
                # print(e_cd.shape)
                
                # antithetic sampling 对偶抽样法
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                x_t_cd, loss_cd = noise_estimation_loss(model_cd, cd, t, e_cd, self.betas)
                
                # Backpropagation and optimization
                optimizer_cd.zero_grad()
                loss_cd.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model_cd.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer_cd.step()
                loss_sum += loss_cd.item()
                average_loss=loss_sum/num
                import sys
                print(f"[epoch: {epoch}/{self.config.training.n_epochs}] ,step: {step}, loss: {loss_sum:.6f}, Average_loss: {average_loss:.6f},data time: {data_time / (i + 1):.4f}", end='\r')
                sys.stdout.flush()
                

                if self.config.model.ema:
                    ema_helper.update(model_cd)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model_cd.state_dict(),
                        optimizer_cd.state_dict(),
                        epoch,
                        step,
                    ]

                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

            print("\nNow,eval!")
            re_datas = []
            count = 0
            all_loss = 0
            for tdata in test_loader:
                count += 1
                import sys
                sys.stdout.write(f"The data is creating: {count}\r")
                sys.stdout.flush()

                model_cd.eval()

                with torch.no_grad():
                    tdata = torch.reshape(tdata, (4, 64, args2.input_size))
                    tdata = tdata.type(torch.FloatTensor)
                    tdata = tdata.to(self.device)
                    #tdata = unet(tdata)
                    z = transz(tdata, 'en')
                    z = z.permute(0, 2, 1)
                    coeffs = dwt1d(tdata)
                    coeffs = dwt1d(z)
                    ca, cd_list = coeffs
                    cd = cd_list[0] 
                    n = tdata.size(0)
                    e_cd = torch.randn_like(cd)
                    b = self.betas
                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    new_ca = ca
                    
                    cd_t, loss_cd = noise_estimation_loss(model_cd, cd, t, e_cd, self.betas)
                    new_cd = self.sample_image(cd_t, 50, model_cd, last=True)
                    new_ca = new_ca.to(self.device)
                    new_cd = new_cd.to(self.device)
                    re_z = idwt1d((new_ca, [new_cd]))
                    re_z = re_z.permute(0,2,1)
                    re_z = re_z.to(self.device)
                    tdata_prime = transz(re_z, 'de')
                    tdata=tdata_prime
                    data = torch.reshape(tdata, (256, args2.input_size)) 
                    re_datas.extend(data)

            re_datas = torch.tensor(np.array([item.cpu().detach().numpy() for item in re_datas]))

            real_data = real_data[:int(len(re_datas))]
            if torch.isnan(real_data).any() or torch.isinf(real_data).any():
                print("real_data contains NaN or Inf")

            if torch.isnan(re_datas).any() or torch.isinf(re_datas).any():
                print("re_datas contains NaN or Inf")

            f1=torch.nn.functional.mse_loss(real_data, re_datas)



            earlyloss = f1
            print('earlyloss={}'.format(earlyloss))
            early_stopping(earlyloss, model_cd,states,'ddim')

            if early_stopping.early_stop:
                print("*******************early stop*********************")
                break
        datalast = time.time()
        print((datalast -datafirst )/60)

       

        if not self.args.use_pretrained:
            model_ca = Model2(self.config)
            model_cd = Model2(self.config)
            if getattr(self.config.sampling, "ckpt_id", None) is None:
               #load model
                states = torch.load(
                    './earlysave/best_newPSM_DMnetwork.pth',
                    map_location=self.config.device)
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )

            model_ca = model_ca.to(self.device)
            model_cd = model_cd.to(self.device)
            model_cd.load_state_dict(states[0])

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model_ca)
                ema_helper.register(model_cd)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model_ca)
                ema_helper.ema(model_cd)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            # model.load_state_dict(torch.load(ckpt, map_location=self.device))
            # model.to(self.device)
            # model = torch.nn.DataParallel(model)

        model_cd.eval()

        if self.args.fid:
            pass
        elif self.args.interpolation:
            pass
        elif self.args.sequence:
            print("sample")
            self.sample_sequence(model_ca, model_cd,transz,unet)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_sequence(self, model_ca, model_cd,transz,unet):

        args, config = self.args, self.config
        dwt1d = DWT1DForward(J=1, mode='zero', wave='db1').to(self.device)
        idwt1d = DWT1DInverse(mode='zero', wave='db1').to(self.device)
        with torch.no_grad():
            print("Start smaple!")
           
            print('sequence')
            if (args.dataset == 'SMAP'):
                print('load SMAP')
                testdata = np.load('./data/SMAP/SMAP/SMAP_test.npy')
                scaler = MinMaxScaler(feature_range=(0, 1))  
                testdata = scaler.fit_transform(testdata)
                label = np.load('./data/SMAP/SMAP/SMAP_test_label.npy')
                testdata = testdata
                label = label


            elif (args.dataset == 'WADI'):
                testdata = pd.read_csv('./data/WADI/WADI_test.csv')
                testdata =testdata.fillna(testdata.mean())
                testdata =testdata.fillna(0)
                testdata = testdata.values[:, 1:-1]
                testdata = testdata.astype(np.float32)
                scaler = MinMaxScaler(feature_range=(0, 1))
                testdata = scaler.fit_transform(testdata)
                label = pd.read_csv('./data/WADI/WADI_label.csv')
                label = label.values[:, :]


            elif (args.dataset == 'SWAT'):
                testdata = pd.read_csv('./data/SWAT/SWaT_test.csv')
                testdata =testdata.fillna(testdata.mean())
                testdata = testdata.values[:, 0:-2]
                testdata = testdata.astype(np.float32)
                scaler = MinMaxScaler(feature_range=(0, 1))  
                testdata = scaler.fit_transform(testdata)

                
                label = pd.read_csv('./data/SWAT/SWaT_test.csv')
                label=label.values[:,-1]

               #label = np.load('./data/SWAT/SWaT_labels.npy').astype(float)
            elif (args.dataset == 'PSM'):
                testdata = pd.read_csv('./data/PSM/PSM/test.csv')
                testdata =testdata.fillna(testdata.mean())
                testdata = testdata.values[:, 1:]
                testdata=testdata.astype(np.float32)
                scaler = MinMaxScaler(feature_range=(0, 1))  
                testdata = scaler.fit_transform(testdata)
                label = pd.read_csv('./data/PSM/PSM/test_label.csv')
                label = label.values[:, 1:]


            label = label.astype(None)
            label = torch.Tensor(label)
            real_data = torch.Tensor(testdata)
            # testdata = testdata[:1280]
            # label = label[:1280]
            #window data
            # windowsize = 64
            # stride = 1

            #testdata = get_from_one(testdata, window_size=windowsize, stride=stride)

            dataloader = DataLoader(
                testdata, batch_size=256, shuffle=True, num_workers=16, drop_last=True,
                pin_memory=True)



            

            re_datas = []
            i = 0
            #Different step
            ts = [50]
            for tt in range(len(ts)):
                for data in dataloader:
                    import sys
                    sys.stdout.write(f"Now step = {ts[tt]}, The data is creating: {i}\r")
                    sys.stdout.flush()

                    i += 1
                    data = torch.reshape(data, (4, 64, args2.input_size))
                    data = data.type(torch.FloatTensor)
                    data = data.to(self.device)
                    #data = unet(data)

                    z = transz(data, 'en')

                    z = z.permute(0, 2, 1)
                    coeffs = dwt1d(z)
                    ca, cd_list = coeffs
                    cd = cd_list[0]

                    n = z.size(0)
                    e_cd = torch.randn_like(cd)
                    b = self.betas
                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=ts[tt], size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    new_ca = ca
                    cd_t, _ = noise_estimation_loss(model_cd, cd, t, e_cd, self.betas)
                    new_cd = self.sample_image(cd_t, ts[tt], model_cd, last=True)
                    new_cd = new_cd.to(self.device)
                    re_z = idwt1d((new_ca, [new_cd]))
                    re_z = re_z.permute(0, 2, 1)
                    re_z = re_z.to(self.device)
                    data_prime = transz(re_z, 'de')
                    data = data_prime
                    data = torch.reshape(data, (256, args2.input_size))
                    re_datas.extend(data)

            re_datas = torch.tensor(np.array([item.cpu().detach().numpy() for item in re_datas]))
            # re_datas = get_from_all(re_datas,window_size=64,stride=1)
            # re_datas = torch.tensor(re_datas)
            print("The length of the data is {}".format(len(re_datas)))
            label = label[:int(len(re_datas)/len(ts))].cpu().detach().numpy()
            real_data = real_data[:int(len(re_datas)/len(ts))].cpu().detach().numpy()
            re_datas = re_datas.cpu().detach().numpy()
            torch.save(re_datas, 're_datas_SMAP_v13-R.pt')
            if len(ts) == 1:
                mind = None
            else:
                mind = [0,0,0,1]
            
           

            metrics_calculate(real_data, re_datas, label,mind=mind)




    def sample_image(self, x,t_1, model, last=True):

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":

                skip = self.num_timesteps // self.args.timesteps


                seq = range(0, t_1, skip)

            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs= generalized_steps(x, seq, model, self.betas, eta=self.args.eta)

            x = xs
        elif self.args.sample_type == "ddpm_noisy" or self.args.sample_type == "ddim":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            # from functions.denoising import ddpm_steps

            # x = ddpm_steps(x, seq, model, self.betas)
            if self.args.sample_type == "ddim":
                  from functions.denoising import ddim_steps
                  x = ddim_steps(x, seq, model, self.betas)
            else:
                  from functions.denoising import ddpm_steps
                  x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:

            x = x[0][-1]

        return x

    def test(self):
        pass
