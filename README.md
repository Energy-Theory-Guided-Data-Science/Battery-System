# Battery-System
Can domain knowledge help to model a battery system?

## Getting Started
In order to get this project up and running on your machine there are several steps to be taken
### 1. Include Data
The data used for analysis needs to be included separately
1. You can download it [here](https://publikationen.bibliothek.kit.edu/1000094469)

After extracting the zip, you should be provided with a folder named `fobss_data`

2. Move it into `Battery-System/data/raw/`

Now you are all set! Navigate to `Battery-System/notebooks/data/fobss_overview.ipynb` to get an overview of the data

### 2. Install Environment
In order to be able to run the provided jupyter notebooks there are several dependencies needed to be installed. Using an anaconda shell please enter:

`conda env create -f environment.yml`

This command will install all necessary dependencies needed for this project in an environment called "battery-system". Activate this environment using:

`conda activate battery-system`

Execute the following command inside the activated environment to create a new kernel:

`python -m ipykernel install --user --name battery-system --display-name "battery-system"`

Now the installation should be completeted and all necessary dependencies should be included. 

## Data
The data we used in our approach is the FOBSS (Frequent Observations from a Battery System with Subunits) dataset. The data was conducted on a battery system with several battery packs each monitored by a subunit of the Battery Management System (BMS). They monitored the current, voltage and temperature during several charge, discharge and rest procedures, called profiles. 

The measurements were taken on three different levels of the system: 
- Inverter: responsible for charging and discharging the battery
  - Current & Voltage
- BMS Master: responsible for monitoring the overall state of the battery
  - Current & Voltage
- BMS Slaves: responsible for monitoring the cells within one battery pack
  - Temperature & Voltage

## Useful Materials
* [How to Write a Good Git Commit Message](https://chris.beams.io/posts/git-commit/)
* [Python Styleguide by Google](http://google.github.io/styleguide/pyguide.html)
