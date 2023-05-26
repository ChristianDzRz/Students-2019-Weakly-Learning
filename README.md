# Extracting strong labels from weakly labeled data (PWR)

The objective is to evaluate different models for the large-scale detection of sound evens using weakly labeled data. The system should provide not only the event class but also its onset and offset. 

## Table of contents
* [Description](#description)
* [Audio dataset](#audio-dataset)
* [Installation](#installation)
* [Usage](#usage)
## Description

The project is inspire in the DCASE2018 Challenge: Large_scale weakly labeled semi-supervised sound event detection in domestic environment

In our system we implemented three models: 
```
Hidden Markov Model
Deep gradient (XGBoost)
Convolutional Recurrent Neural Network
```
In order to find the one with higher performance, each model will be evaluated using a Evaluation toolbox for Sound Event Detectection
[sed_eval](https://github.com/TUT-ARG/sed_eval)

## Audio dataset

The dataset used was obtain from the DCASE2018 Challenge too. The audioset consist of 10 classes of sound events:

* Speech             
* Dog 
* Cat 
* Alarm/bell/ringing 
* Dishes 
* Frying 
* Blender 
* Running water 
* Vacuum cleaner 
* Electric shaver/toothbrush 


The data is splitted into two sets:
```
Training set:
    -Labeled: Verified weak annotations.              
    -Unlabeled in domain: Distribution per class is close to labeled one.
    -Unlabeled out of Domain: 
Test set: Strong labeled and with similar distribution in terms of clips per class to the weakly labeled training set.
```

In order to simplify its implementation its been uploaded to [Google Drive](https://drive.google.com/open?id=1gl-z3eSUVNKrphdPXInPBCja7dMVSDxT)

More information from the dataset can be found in [here](https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task4/dataset)


## Instalation
```
1.  Clone the repository using git clone https://gitlab.com/zkwiatkowska/students-2019-weakly-learning.git inserting your gitlab credentials.
2.  Install the requirements using: pip install -r requirements.txt.
3.  Download the dataset
4.  Modify in config.yaml the path in which the dataset is stored (db_path: 'your_path')

```
Its recommended to create a virtualenv before installing the requirements, this can be done by:
```
1.  Opening terminal in your project folder
2.  Installing virtualenv tool using: pip install virtualenv
3.  Create the environment using: virtualenv <name_of_the_enviroment>
```

## Usage

The desired settings have to be defined in the config.yaml

Defining the sampling rate:
```
sample_rate: <sampling_rate_value> 
```

Selecting the model:
```
Hidden Markov Model
    model_name: 'HMM'

Deep gradient (XGBoost)
    model_name: 'GradientLearning'

Convolutional Recurrent Neural Network
    model_name: 'CRNN'
```
In order to select the desired dataset:

Specifying the operations:

```
Opertion:
    load_config: Change the settings of the system
    read_data: Load the data from the db_path\audio\test folder into the model
    generate_submission: Model generates a prediction according to the sample used (right now is random)
    evaluate: Evaluate the system performance

They should be added into the array in the order you wish to execute them, for example:
    ops : ['load_config', 'read_data' ,'generate_submission']

```

Run ```main.py```



