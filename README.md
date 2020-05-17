# wavenet\_autoencoders
This is a submission to [ZeroSpeech 2020 challenge](https://zerospeech.com/2020/results.html).

This work is based on [Chorowski' wavenet autoencoder model](https://arxiv.org/abs/1901.08810) and [wavenet vocoder implementation](https://github.com/r9y9/wavenet_vocoder)

This work consits of two models: 
 * WaveNet autoencoder + Instance Normalization (IN-WAE)
 * WaveNet autoencoder + Sliced Vector Quantization (SVQ-WAE)
 
![Model](AE.png)

# Requirements
 1 Python 3.6
 2 PyTorch 0.4.1
 3 tensorboardX
 4 challenge evaluation scripts [rep](https://github.com/bootphon/zerospeech2020)

# Steps to run

## Download data
`bash ./run_bash/download_dataset.sh`

Unzip the dataset requires 7z (>16.04) and password
