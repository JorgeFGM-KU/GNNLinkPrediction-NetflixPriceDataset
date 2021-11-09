# The Repository

This repository contains code written by:
 - 김혜림
 - 오형락
 - 이재후
 - Gimenez Perez Jorge

Students of the Machine Learning course (Fal 2021) at Korea Univeristy.  
The aim of this project is to build a Graph Neural Network able to predict the rating a person would give to a movie by using GrpahSAGE model on the Netflix Price Dataset.  

# The Folders and Files

- **data/**: All the Netflix Price Dataset related data files can be found here.
  - **processed/**: -Nothing here yet-
  - **raw/**: Raw data files as obtained from Kaggle can be found in this folder.
  - **toy_dataset/**: Data files for the toy_dataset geerated to speed up the model testing and building when developing can be found in this folder.
    - **processed/**: -Nothing here yet-
    - **raw/**: A small *simulation* of the Netflix Price Dataset can be found in this folder. This file can be used to speed up the building and testing of the model when developing.
- **src/**: All the Python code can be found in this folder.
  - **data_preprocessing.py**: Functions related to data preprocessing can be found in this file. Currently only the `get_dataframe_from_dataset` function is defined and implemented.
  - **generate_graphs.py**: Functions related to the generation of the graph objects required by graph neural networks libraries can be found in this file. Currently functions `generate_stellar_graph` and `generate_dgl_graph` which will generate the necessary graph objects for StellarGraph and DGL libraries correspondingly are defined but not implemented.
  - **generate_toy_dataset.py**: This program can be run in order to generate a new toy dataset that emulates the real Netflix Price Dataset. There are some parameters that can be tweaked inside this file in order to obtain different toy datasets. The generated file that this program produces can be found at `data/toy_dataset/raw/`.
  - **main.py**: This fail contans the main program that should be run, it will call all the necessary dependencies.
