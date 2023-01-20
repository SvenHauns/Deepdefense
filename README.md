
# Deepdefense: Automatic annotation and classification of immune systems in prokaryotes using Deep Learning


Deepdefense is a tool to serach for immune system cassettes based on the Doron system. To achieve this 
Deepdefense uses two distinct prediction modules. The first module consists of an ensemble of Deep Learning models to 
reject proteins, that do not belong to any immune system. The second module uses an ensemble of Deep Learning models
to classify proteins. We use a Deep Open Classifier (DOC) together with temperature scaling to calibrate our models.
The calibration allows us to either 1) reject proteins 2) classifiy them as a known type or 3) classify them as
an unknown, potentially new type of protein.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

First you need to install Miniconda
Then create an environment and install the required libraries in it


### Creating a Miniconda environment 

First we install Miniconda for python 3.
Miniconda can be downloaded from here:

https://docs.conda.io/en/latest/miniconda.html 

Then Miniconda should be installed. On a linux machine the command is similar to this one: 

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Then we create an environment. The necessary setup is provided in the "environment.yml" file inside the "for_environment" directory

In order to install the corresponding environment one can execute the following command from the "for_environment" directory

```
conda env create -f environment.yml
```



### Activation of the environment

Before running CRISPRidentify one need to activate the corresponding environment.

```
conda activate crispr_identify_env
```



## Running Deepdefense


### Models


Due to the file size restrictions of github, models are available on:

https://drive.google.com/file/d/1sEBtVYExPIl47-YmkNTzVDDU42aFA7vb/view?usp=sharing
https://drive.google.com/file/d/1Obek8fj2G67UeDVN-95Em_msagcgb1qO/view?usp=sharing


### Training custom models

if you want to train your own custon models based on the Deepdefense architecture and training procedure, you can use:

```
python main.py -f <train_file_class_1> -f <train_file_class_2> -t <test_file_class_1> -t <test_file_class_2> -v <val_file_class_1> -t <val_file_class_2>
               -p <path_to_models> -n <architecture> 


```

additionally we provide flags for:


* `--calibration_method <calibration_method>`

* `--epochs <epochs>`

* `--batch_size <batch_size>`




### creating the output

In the first step we create and save the prediction for both ensembles of models. This makes rerunning the cassette prediciton later on less time consuming.

To create the model predictions run:


```
python predict_and_save.py -f <data_folder> -r <results_folder>

```

The results will be save as a pickle file in <results_folder>.




additionally we provide flags for:


* `--prodigal <prodigal>`

* `--prodigal_completeness <prodigal_completeness>`




### creating the cassette prediction

To predict the cassettes run the command below with the output of the step above. This creates the output file.

```
python predict_cassette.py -f <data_folder>

```

additionally we provide flags for:


* `--output_file_name <output_file_name>`

* `--max_distance <max_distance>`

* `--min_correspondence <min_correspondence>`

max_distance regulates the maximum distance between detected genes allowed.
min_correspondence regulates the minimum number of classes that need to be detected to allow for classification of the cassette.


#### Interpreting the output

The output details the file, the rules found, the indeces of the proteins used for the prediction.



## Improving Deepdefense

We are constantly working on the improvements of Deepdefense. If you found a bug please submit via github issue interface.





