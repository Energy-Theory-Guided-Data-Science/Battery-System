# Setup
To be able to execute the notebooks in this directory, three steps have to be taken:

## 1. Initialize General Hyper Parameters
Run the notebook `00-Data_Setup.ipynb` to initialize the hyper parameters needed for every model.

## 2. Create Thevenin Model
Run the notebeook `theory-baseline/01-T_Thevenin Model.ipynb` to create a Thevenin model since this is needed by the majority of theory-guided models.

## 3. Select Thevenin Model
The newly created Thevenin model now has to be set in the respective model that requires it. Follow these steps analogously for all theory-guided models that require a Thevenin model:

3.1 When creating the Thevenin model, cell 7 should print something like `Model saved to /models/T/theory_baseline-2336...` (2336 is just an example, this number will be different for you)

3.2 If you want to run e.g. the notebook `theory-guided/03-TGDS_Initialization.ipynb` you have to modify the model_hyperparameters dictionary. For the key `theory-model`, please enter the number that you saw in 3.1 (in our case 2336).
