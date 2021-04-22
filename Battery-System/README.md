Battery-System
==============================


Project Organization
------------

    ├── LICENSE                 <- MIT License
    ├── README.md               <- The top-level README for developers using this project
    ├── data                    <- Stores all necessary data
    │   ├── processed           <- Additional data needed by the models
    │   └── raw                 <- The original, immutable data dump
    │
    ├── models                  <- Trained and serialized models
    │   ├── DS                  <- Data Baseline models
    │   ├── T                   <- Theory Baseline models
    │   └── TGDS                <- Theory-guided models
    │
    ├── notebooks               <- Jupyter notebooks
    │   ├── data                <- To visualize the FOBSS dataset
    │   │   └── profiles        <- Plots of all profiles in the FOBSS dataset
    │   │
    │   ├── features            <- Different analysis of the FOBSS dataset
    │   │
    │   └── models              <- Several Notebooks which execute the source code providede in src/
    │       ├── data_baseline   <- Data Science based models
    │       ├── theory_baseline <- Theory based models
    │       └── theory_guided   <- Models combining theory with Data Science
    │
    ├── reports                 <- Generated graphics to be used in reporting
    │   └── figures             <- Learning curves during training as well as plots of the test data profiles 
    │   
    ├── src                     <- Source code for use in this project
    │   ├── data                <- Utility functions to preprocess the raw data for later use 
    │   │
    │   └── models              <- Scripts which capsulate the models trained and executed in the jupyter notebooks

--------
