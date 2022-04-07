# wavenet\_autoencoders
This is a submission to [ZeroSpeech 2020 challenge](https://zerospeech.com/2020/results.html).

This work has been accepted by Interspeech2020, paper can be found [here](http://arxiv.org/abs/2008.06892)

This work is based on [Chorowski' wavenet autoencoder model](https://arxiv.org/abs/1901.08810) and [wavenet vocoder implementation](https://github.com/r9y9/wavenet_vocoder)

This work consits of two models: 
 * WaveNet autoencoder + Instance Normalization (IN-WAE)
 * WaveNet autoencoder + Sliced Vector Quantization (SVQ-WAE)
 
![Model](AE.png)

# Requirements
 1. Python 3.6
 2. PyTorch 0.4.1
 3. tensorboardX
 4. challenge evaluation scripts [rep](https://github.com/bootphon/zerospeech2020)
 5. librosa
 6. scipy

# Steps to run

## Download data
`bash ./bin/download_dataset.sh`

Unzip the dataset requires 7z (>16.04) and password

## Preprocessing
`bash ./bin/run_pre.sh 2020/2019`

## Train
 * train VQ-WAE model
    
    `bash ./bin/vqwae_train.sh exp_name hps language`
    
    e.g. `./bin/vqwae_train.sh exp_name hps/vqwae.json english`
 

