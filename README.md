# Twavefussion
TWavefussion: Wavelet-based Diffusion with Transformer for Multivariate Time Series Anomaly Detection

This repository contains code for the paper, TWavefussion: Wavelet-based Diffusion with Transformer for Multivariate Time Series Anomaly Detection.
(The code is being sorted out and we will continue to update it.)

##  Overview

In this paper, TWavefussion, an unsupervised diffusion model for MTS anomaly detection combining wavelet-based diffusion model and transformer autoencoder, is proposed. The wavelet-based diffusion model captures fine-grained local features in the high-frequency components of latent features and helps fuse both local and global MTS features better. Comparative experiments show TWavefussion achieves leading performance on three of four datasets.

## Datasets

1. PSM (PooledServer Metrics) is collected internally from multiple application server nodes at eBay.
2. SMAP (Soil Moisture Active Passive satellite) also is a public dataset from NASA. 
3. WADI (Water Distribution) is obtained from 127 sensors of the critical infrastructure system under continuous operations. 
4. SWAT (Secure Water Treatment) is obtained from 51 sensors of the critical infrastructure system under continuous operations. 

We apply our method on four datasets, the SWAT and WADI datasets, in which we did not upload data in this repository. Please refer to [https://itrust.sutd.edu.sg/](https://itrust.sutd.edu.sg/) and send request to iTrust is you want to try the data.

## How to run

- Train and detect:

> CUDA_VISIBLE_DEVICES = {gpu_id} python main.py  --config test.yml  --doc ./{dataset}  --sequence

For example: CUDA_VISIBLE_DEVICES = 0 python main.py  --config test.yml  --doc ./PSM  --sequence

Then you will train the whole model and will get the reconstructed data and detected score.

## How to run with your own data

- By default, datasets are placed under the "data" folder. If you need to change the dataset, you can modify the dataset path  in the main file.Then you should change the corresponding parameters of TIMEEMB2.py and diffusion.py

> python main.py  --'dataset'  your dataset

## Result

We  use dataset PSM for testing demonstration, you can run main.py directly and get the corresponding result.

## References
Our implementation is based on [TimeADDM](https://github.com/Hurongyao/TIMEADDM). We would like to thank them
