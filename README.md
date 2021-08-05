# Continuous-HAR


### Install Dependencies
```shell
conda create -n pytorch python==3.7
conda activate pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge scikit-learn
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorboardx
conda install -c conda-forge tqdm
```

### Dataset downloading
```shell
mkdir datasets
cd datasets
mkdir discrete
cd discrete
wget http://researchdata.gla.ac.uk/848/20/Dataset_848.7z
```
unzip the discrete and continuous data files

### Preprocess raw data to extract Doppler
```shell
cd ..
mkdir processed
cd ..
```
run `preprocess.m` on both discrete and continuous data
```shell
mkdir data
python data_prep.py
```



### Train on discrete data to get Doppler embeddings

### Train on continuous data