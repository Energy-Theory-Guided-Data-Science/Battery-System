# Profiles
The FOBSS profiles are described in great detail in data/fobss_data/profiles.xlsx. This README should provide a classification of the different profiles to help with identifying possible use cases for training the model. Some of these profiles are repeated in different runs indicated by the number in the parantheses:
## Small profiles
These profiles are short in time and at most include 3 charge or discharge processes. 
- -10A 25A (1)
- -10A -25A (1)
- -10A (9)
- 25A (8)
- 10A (10)
- 10A 3x (1)
- 10A -10A (4)
- 10A 25A (1)
- -25A (7)
## Long Runs
These profiles are long and include multiple charge and discharge processes repeating after a certain interval
- -25A 10 A (1)
- ocv_soc (1)
- Ri Jumps (1)
- Ri Jumps 25A (4)
## Complex profiles
These profiles are short in time but complex with respect to the current curve
- current_ramp (1)
- stairs (1)
- osz (1)
- osz_small (1)
