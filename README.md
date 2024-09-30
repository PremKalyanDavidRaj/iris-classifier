 Iris Image Classification

## Purpose
- Consolidate understanding of how to build an image classifier using Keras's `Sequential` model
- Become familiar with development in the cloud of SageMaker Studio Lab 
- Practice the use of Copilot Chat to understand core concepts and techniques illustrated in the lab. 

## Guidelines
- See **SageMaker Studio Lab Development** resource in the **Resources** module in Canvas. It helps you:
  - Learn what SMSL `conda` environments are set up by default
  - Deactivate the current environment to get ready to set your lab3 environment.
- Clone the remote repo **lab3** from GitHub to your SageMaker Studio Lab (SMSL) account in **comp841** directory.
- Create a new virtual environment, `irisenv`, that has a Pythohn 3.11 instance:
```
conda create -n irisenv python=3.11
conda env list
conda activate irisenv
conda list
```
- Install `ipykernel` package
```
conda install ipykernel
```
- Install three other packages: `pandas`, `scikit-learn`, and `tensorflow`:
```
pip install pandas
pip install scikit-learn
pip install tensorflow
```
- Verify the installation: `conda list`


