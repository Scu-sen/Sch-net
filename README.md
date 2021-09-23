

This repo contains the source code for automatic detection

## Quick setup and start 

#### Software

* Linux
* Nvidia drivers, CUDA >= 10.0, cuDNN >= 7


#### Hardware

* 32GB of RAM
* 2080ti or another GPU with fp16 support and at least 12GB memory 

### Preparations
Download and extract [dataset](https://figshare.com/articles/dataset/New_draft_item/2360626) 
Generate csv

```
    filename,label
    
    ./data/1,1
    ./data/2,0 
```
 ### Run
 python train_main.py

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://doi.org/10.1186/s12938-021-00915-2):

```@article{WANG2021102785,
title = {Sch-net: a deep learning architecture for automatic detection of schizophrenia},
journal = {BioMedical Engineering OnLine},  
pages = {1-21},  
year = {2021}  
```
