# Time-Aware-Location-Prediction
This work "Time-Aware Location Prediction by Convolutional Area-of-Interest Modeling and Memory-Augmented Attentive LSTM" has been published in TKDE 2022.
## :page_facing_up: Description
t-LocPred is a novel time-aware location prediction model for Point of Interests (POIs) recommendation. It consists of a convolutional AoI modeling module (ConvAOI) and memory-augmented attentive LSTM (mem-attLSTM). It captures both coarse- and fine-grained spatiotemporal correlations among a user’s historical check-ins and models his/her long-term movement patterns. 
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- NVIDIA GPU (RTX 3090) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/Time-Aware-Location-Prediction.git
    cd Time-Aware-Location-Prediction
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :cd: Data Preparation
1. You can modify the config file [one_day_data_general/conf.py](https://github.com/BIT-MCS/Time-Aware-Location-Prediction/blob/main/one_day_data_general/conf.py) for data preparation.
For example, you can control the length of check-in sequences by modifying this line:
	```
	[41]  'seq_len': 8,
	```
2. Using the following commands to process the original datasets and generalize the data for t-LocPred.
	```bash
	cd one_day_data_general
	python main.py
	```
## :computer: Training

We provide complete training codes for t-LocPred.<br>
You could adapt it to your own needs.

1. If you don't have NVIDIA RTX 3090, you should comment these two lines in file [utils.py](https://github.com/BIT-MCS/Time-Aware-Location-Prediction/blob/main/step2model/utils.py).
	```
	[19]  torch.backends.cuda.matmul.allow_tf32 = False
	[20]  torch.backends.cudnn.allow_tf32 = False
	```
2. You can modify the config files 
[step1model/ConvAOI/conf.py](https://github.com/BIT-MCS/Time-Aware-Location-Prediction/blob/main/step1model/ConvAOI/conf.py).<br>
For example, you can control the hyperparameter about CNN kernal size in convolutional AoI modeling module by modifying this line:
	```
	[31]  'cnn_kernel_size': 3,
	```
3. Training the ConvAOI module:
	```
	cd step1model/ConvAOI
	python main.py
	```
4. Training the mem-attLSTM module:
	```
	cd step2model/mem-attLSTM
	python main.py
	```
## :checkered_flag: Testing
1. Running the ConvAOI module:
	```
	cd step1model/ConvAOI
	python test.py
	```
2. Running the mem-attLSTM module:
	```
	cd step2model/mem-attLSTM
	python test.py
	```
## :e-mail: Contact

If you have any question, please email `2656886245@qq.com`.
## Paper
If you are interested in our work, please cite our paper as

```
@ARTICLE{9128016,
  author={Liu, Chi Harold and Wang, Yu and Piao, Chengzhe and Dai, Zipeng and Yuan, Ye and Wang, Guoren and Wu, Dapeng},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Time-Aware Location Prediction by Convolutional Area-of-Interest Modeling and Memory-Augmented Attentive LSTM}, 
  year={2022},
  volume={34},
  number={5},
  pages={2472-2484},
  doi={10.1109/TKDE.2020.3005735}
}
```
