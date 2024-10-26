# Research on Cerebral Cortex Classification Using Data Augmentation

## Project Overview
This project aims to classify the cerebral cortex using data augmentation techniques. By leveraging a Multi-Scale Adversarial Regularization Autoencoder (MSVGAE) and other machine learning models, we enhance classification performance.

## File Structure
- `train-MSVGAE.py`: The main script to train the MSVGAE model.
- `MSVGAE.py`: Core implementation of the Multi-Scale Adversarial Regularization Autoencoder.
- `MSVGAE_Encoder.py`: Defines the encoder layer of the MSVGAE.
- `NetRA.py`: Contains the architecture definitions for MLP_G and MLP_D.
- `load_data.py`: Script for loading the dataset.
- `preprocessing.py`: Data preprocessing module, including data cleaning and transformation.

## Requirements
- Python 3.x
- Required libraries:
  - numpy
  - pandas
  - pytorch (depending on your needs)
  - scikit-learn
  - etc.

## Usage Instructions
1. Ensure that the required Python libraries are installed.
2. Place the dataset in the specified path (refer to `load_data.py` settings).
3. Run the main script:
   ```bash
   python train-MSVGAE.py

