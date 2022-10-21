# MANAS
This repository includes the implementation for Learning Basic Skills and Reuse: Modularized Adaptive Neural Architecture Search (MANAS):

*Hanxiong Chen, Yunqi Li, He Zhu, Yongfeng Zhang. 2022.*

## Refernece

For inquiries contact Hanxiong Chen (hanxiong.chen@rutgers.edu) or Yongfeng Zhang (yongfeng.zhang@rutgers.edu)

## Environments

```
Python>=3.6
numpy>=1.18.1
pytorch>=1.4.0
pandas>=1.1.0
scipy>=1.3.0
tqdm>=4.32.1
scikit_learn>=0.23.1
```

## Datasets

- The processed datasets are in  [`./dataset/`](https://github.com/TalonCB/MANAS/tree/master/dataset)

- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 
    

## Example to run the codes
-   To guarantee the program can execute properly, please keep the directory structure as given in this repository. The parameter configuration file is located at [`./src/config.py`](https://github.com/TalonCB/MANAS/tree/master/src/config.py)
-   Example run:

```
# MANAS on Cellphone dataset
> cd ./src/
> python ./main.py --dataset Cellphone --child_model_path "../model/Cellphone_child_model.pt" --controller_model_path "../model/Cellphone_controller_model.pt"
```

