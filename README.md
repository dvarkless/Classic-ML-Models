 
<div align="center">

# Classic ML models

Custom command line interface for Wallpaper Engine for KDE widget

[Setup](#Setup-and-run) •
[Usage](#usage) •
</div>

## About The Project
This project serves an educational purpose of showing how models in 
Machine Learning work under the hood. I've made them using exclusively Python
code library and numpy.  
Where are several classification models:  
- Logistic Regression
- K nearest neighbors
- Support Vector Machine
- Bayesian Classificator
- Decision tree


### Prerequisites
`Python >= 3.9`  
`Jupyter Notebook` to run code in the notebook

### Setup and run
1. Clone the repository by running
```
git clone https://github.com/dvarkless/Classic-ML-Models.git
```    
2. Create a python virtual environment:
```
cd Classic-ML-Models
python -m venv venv
```   
3. If you are using Linux or Mac:
```
source ./venv/bin/activate
```  
If you are using Windows:
```
./venv/Scripts/activate.ps1
```  
4. Create an IPython kernel if you want to run it in Jupyter Notebook:   
```
python -m ipykernel install --user --name=classic-ml-models
```

## Usage
1. Prepare a dataset, split it into training data, evaluation input and evaluation answers:  
python```
training_data = np.genfromtxt("datasets/light-train.csv", delimiter=",", filling_values=0)
evaluation_data = np.genfromtxt("datasets/medium-test.csv", delimiter=",", filling_values=0)
evaluation_input = evaluation_data_lite[:, 1:]
evaluation_answers = evaluation_data_lite[:, 0]

datapack = (training_data, evaluation_input, evaluation_answers)  
```
2. Pass hyperparameters into two dicts:
- The first one is used to create a class instance
- The second dict passes parameters into model one-by-one. It is used to show 
how different parameters affect the model's prediction quality
python```
hp = {
    'data_converter': get_plain_data,
    'normalization': True,
    'shift_column': True,
    'learning_rate': 0.05,
    'batch_size': 300,
    'epochs': 300,
    'num_classes': 26,
    'reg': 'l1',
    'reg_w': 0.05,
}

params_to_change = {
    'learning_rate': [0.01, 0.02, 0.05],
}
```
3. Run the model using a ModelRunner class:  
```
MultilogRunner = ModelRunner(MultilogRegression, defaults=hp, metrics=my_metrics, responsive_bar=True)
MultilogRunner.run(*datapack, params_to_change, one_vs_one=True)
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
