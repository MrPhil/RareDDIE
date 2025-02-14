# Predicting rare drug-drug interaction events with dual-granular structure-adaptive and pair variational representation
This is a meta-learning-based DDIEs predictor. Before the article is published, this project only contains all data,  and the reproduction code is shown in the submission document.



## System requirements
Installation Tested on Ubuntu 16.04, CentOS 7, windos 10 with Python 3.7 on one NVIDIA RTX 4080Ti GPU.



# Installation 
After downloading the code and data, execute the following command to install all dependencies. This may take some time.

```bash
pip install -r requirements.txt
```



# Quick Demo & Instructions for use
The repositories for **Independent**, **RareDDIE** and **ZetaDDIE** provide code for reproducing our results.

### **Result Reproduction**

- Run `tester_struc_drugbank.py` and `tester_struc_mdf.py` to reproduce the reported results.

### **Model Training**

- Run `trainer_structure_acc_fp_neigh_VAE_GAN_struc.py` to train the model on the standard dataset1.
- Run `trainer_structure_acc_fp_neigh(mdf)_VAE_GAN_struc.py` to train the model on the standard dataset2.



# Preprocessing personal data sets

Users can preprocess their own datasets for use with our models. A step-by-step example is provided in the *toy example* directory. 

### **Data Preparation**

Users should first prepare their dataset, including interaction event data, drug data, and SMILES representations of drugs. The expected formats follow those in `toy.data`, `druglist.csv`, and `drug_smiles.csv`.

### **Pipeline Execution**

1. **Generate interaction event input files**

   - Run 

     ```
     1construct_task.py
     ```

      to generate event input files:

     - `train_tasks.json`: Common events for training
     - `dev_tasks.json`: Common events for validation
     - `test_tasks.json`: Fewer events for testing
     - `test2_tasks.json`: Rare events for testing

2. **Generate DDIE relationship input files**

   - Run 

     ```
     2data_(get_e1rel_e2_and_rel2candidates).py
     ```

      to produce:

     - `e1rel_e2.json`: Drug-drug interaction event relationships
     - `rel2candidates.json`: Relationship of candidates

3. **Integrate the background graph**

   - Replace the default `path_graph` with a prebuilt background graph.
   - Add `dti_entity.csv` and `dti_rel.csv` to define entities and relationships in the graph.
   - Run `3add_entity_and_rel.py` to incorporate these into the dataset.

4. **Generate drug feature representations**

   - Copy the prepared SMILES file to `fp/data/`.
   - Run `save_features.py` to generate the feature file `morgan_toy_dataset.npz` in the `features` directory.



# Training

A training example for **RareDDIE** is provided in the *toy example* directory.

Run

```
python trainer_structure_acc_fp_neigh_VAE_struc.py --dataset toy_dataset --few 10 --train_few 10 --batch_size 256
```

- If users have pre-trained background graph embeddings (e.g., `DRKG_TransE_entity.npy` and `DRKG_TransE_relation.npy`), they should construct `ent2embids` and `relation2embids` files to map all dataset and background entities/relations to feature indices.
- For entities or relations without pretrained features, set the corresponding index to `-1`.

Run

```
python trainer_structure_acc_fp_neigh_VAE_struc.py --dataset toy_dataset --few 10 --train_few 10 --batch_size 256 --random_embed False
```

To run **ZetaDDIE**, simply replace the preprocessed dataset directory with the appropriate data.



# Test

A test example for **RareDDIE** is provided in the *toy example* directory.

Run

```
python tester_struc_dataset.py
```



# Independent dataset testing

Users can also leverage a model trained on their own standard dataset to directly predict on an independent dataset, enabling cross-domain prediction. We provide code for reproducing our results and user own dataset.

### **Independent Dataset Prediction**

1. **Prepare the independent dataset** following the same preprocessing steps described earlier. Copy the processed dataset into the `Independent` directory (e.g., `twoside`).
2. **Copy the trained standard dataset and trained model** into the `Independent` directory (e.g., `dataset1` and `models`).
3. Execute the prediction script to evaluate the model’s cross-domain performance.

Run

```
python tester_cross_domain.py
```

