# Overview

This repository contains a collection of notebooks and data for the analysis and modelling of investor risk to meet capital calls.

# Structure

Brief overview of key sub-folders:
* `data_reference` - external data files downloaded that are used as reference for data sampling or synthetic generation
* `notebooks` - Jupyter notebooks for sampling, cleaning, labelling and modelling of data based on US SCF
* `notebooks_arc` - Jupyter notebooks used for synthetic data generation
* `sample_data` - data that has been sampled from US SCF and used in notebooks
* `src` - python scripts that are referencing in notebooks

# Steps

1. Prepare the sample dataset using `Sample Data Generation.ipynb` from the US SCF dataset
2. Label the dataset using the Red Flags approach using `Sample Data Cleaning.ipynb`
3. Train and test ML models on the dataset using the One Qualifier Approach using `Sample Data Modelling.ipynb`
4. Build an ensemble model using the Aggregate Qualifier Approach using `Sample Data Ensemble.ipynb`


