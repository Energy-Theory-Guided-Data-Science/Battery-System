
import os 


if __name__ == '__main__':
    
    os.system('python 00-Data_Setup.py')

    for i in range(5):
        print("Experiment Run " +  str(i))
        os.system('python 01-DS_LSTM.py')
        os.system('python 01-T_TheveninModel.py')
        os.system('python 01-Feature_Engineering.py')
        os.system('python 02-TGDS_Loss_Function.py')
        os.system('python 03-TGDS_Initialization.py')
        os.system('python 04-TGDS_Model_Design.py')
        os.system('python 05-TGDS_Hybrid.py')
        os.system('python 06-TGDS_Residual.py')

        
