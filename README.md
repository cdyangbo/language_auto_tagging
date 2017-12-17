# Speech auto tagging
identify the language's attribute(such as language type,gender,accent etc.) use keras.
now only implementation language type classify.

# 1. data prepare
  use librispeech and THCHS30 as english and chinese speech corpus
  
  run in shell:
     python data_prepare
    
# 2. train
  run: python train.py

# 3. test
  run : python test.py
