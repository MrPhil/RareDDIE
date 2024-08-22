# Predicting rare drug-drug interaction events with dual-granular structure-adaptive and pair variational representation
This is a meta-learning-based DDIEs predictor. Before the article is published, this project only contains all data,  and the reproduction code is shown in the submission document.



## System requirements
Installation Tested on Ubuntu 16.04, CentOS 7, windos 10 with Python 3.7 on one NVIDIA RTX 4080Ti GPU.



# Installation 
After downloading the code and data, execute the following command to install all dependencies. This may take some time.

```bash
pip install -r requirements.txt
```



# Demo & Instructions for use
Execute tester_struc_drugbank.py and tester_struc_mdf.py to reproduce the results. Execute trainer_structure_acc_fp_neigh_VAE_GAN_struc.py and trainer_structure_acc_fp_neigh(mdf)_VAE_GAN_struc.py to train the model.