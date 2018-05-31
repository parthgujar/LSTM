# TextPrediction
Text prediction using LSTM

Run with either "train" or "test" mode

# 1. Train mode:

    python words_prediction_lstm.py 'mode' 'data_file' 'model_file' 'max_update' 'regularization' 'learning_rate'  
   
      Mode           : train
      data_file      : training data file
      model_file     : model file (to save optimized paramaters of the model after training)
      max_update     : maximum number of update
      regularization : L1 or L2 or none
      learning_rate  : learning rate of model
   
    e.g: words_prediction_lstm.py train input.txt model_file 5000 L1 0.001

# 2. Test mode:

    python words_prediction_lstm 'mode' 'data_file' 'model_file' 'sample_text' 'newtext_length'
  
      mode              : test
      data_file         : data file (to check new generated text)
      model_file        : model file (saved from train step)
      sample_text       : feed the first text for LSTM to generate new words (by default,
                             the current setting is 3 words. Is is defined as num_input paramater in words_prediction_lstm.py
      newtext_length    : the length of new text, e.g : 50 words
        
    e.g : words_prediction_lstm test input.txt model_file "had a general" 50
