# Battery-System
Can domain knowledge help to model a battery system?


## Project Organization
------------

    ├── LICENSE                       <- MIT License
    ├── README.md                     <- The top-level README for developers using this project
    ├── data                          <- Stores all necessary data
    │   ├── processed                 <- Additional data needed by the models
    │   └── raw                       <- The original, immutable data dump
    │
    ├── models                        <- Trained and serialized models
    │   ├── DS                        <- Data Baseline models
    │   ├── T                         <- Theory Baseline models
    │   └── TGDS                      <- Theory-guided models
    │
    ├── notebooks                     <- Jupyter notebooks
    │   ├── data                      <- To visualize the FOBSS dataset
    │   │   └── profiles              <- Plots of all profiles in the FOBSS dataset
    │   │
    │   ├── features                  <- Different analysis of the FOBSS dataset
    │   │
    │   ├── models                    <- Several Notebooks which execute the source code providede in src/
    │   │   ├── data_baseline         <- Data Science based models
    │   │   ├── theory_baseline       <- Theory based models
    │   │   ├── theory_guided         <- Models combining theory with Data Science
    │   │   └── 00-Data_Setup.ipynb   <- Notebook that needs to be executed before any notebooks in notebooks/models/
    │   │
    │   ├── results                   <- Experimental results as a csv file and a Notebook to create results for the paper
    │   
    ├── src                           <- Source code for use in this project
    │   ├── data                      <- Utility functions to preprocess the raw data for later use 
    │   │
    │   ├── models                    <- Scripts which capsulate the models trained and executed in the jupyter notebooks
    │   │
    │   └── experiments               <- Scripts to run experiments

## Getting Started
In order to get this project up and running on your machine there are several steps to be taken:

### 1. Install Environment
In order to be able to run the provided jupyter notebooks there are several dependencies needed to be installed. Using an anaconda shell in the `Battery-System` folder, please enter :

`conda env create -f environment.yml`

This command will install all necessary dependencies needed for this project in an environment called "battery-system". Activate this environment using:

`conda activate battery-system`

Execute the following command inside the activated environment to create a new kernel:

`python -m ipykernel install --user --name battery-system --display-name "battery-system"`

Now the installation should be completeted and all necessary dependencies should be included. 

### 2. Include Data
The data used for analysis needs to be included separately
1. You can download it [here](https://publikationen.bibliothek.kit.edu/1000094469)

After extracting the zip, you should be provided with a folder named `fobss_data`

2. Move it into `Battery-System/data/raw/`

Now you are all set! Navigate to `Battery-System/notebooks/data/fobss_overview.ipynb` to get an overview of the data.


#### Data
The data we used in our approach is the FOBSS (Frequent Observations from a Battery System with Subunits) dataset. The data was conducted on a battery system with several battery packs each monitored by a subunit of the Battery Management System (BMS). They monitored the current, voltage and temperature during several charge, discharge and rest procedures, called profiles. 

The measurements were taken on three different levels of the system: 
- Inverter: responsible for charging and discharging the battery
  - Current & Voltage
- BMS Master: responsible for monitoring the overall state of the battery
  - Current & Voltage
- BMS Slaves: responsible for monitoring the cells within one battery pack
  - Temperature & Voltage
  
## Running Experiments

1. Run `notebooks/models/00-Data_Setup.ipynb`  in order to initialize the hyperparameters.
2. Run `src/experiments/experiments/Run_experiments.ipynb` to run the experiments.

## Analyze the Results
1. The experimental data results are in `notebooks/results/experimental_results.csv`
2. Run `notebooks/results/Analyze_results.ipynb` to analyze the results used in the paper.

--------
