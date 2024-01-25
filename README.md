There are two stages in training: self supervised pre-training and supervised fine tuning. Initially, self supervised pre-training should be performed and then supervised fine tuning should be done. 

The folder Pre_Train contains the file pre_train.py which is responsible for performing self supervised pre-training. Run this file and after the training, the model will be saved in the same folder.

The folder Fine_Tune contains the file fine_tune.py which is responsible for performing supervised fine tuning. Before running this file, copy all the saved model files from the Pre_Train folder to this folder. After training, the refined model will be saved in the same folder.

The folder Test contains the file test.py which is responsible for the test case inference. Before running this file, copy all the saved model files from the Fine_Tune folder to this folder. 

Paper Link: https://link.springer.com/chapter/10.1007/978-3-031-00123-9_32
