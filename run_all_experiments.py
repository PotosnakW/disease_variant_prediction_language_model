import os

os.system('python run_vespa.py')
os.system('python run_EVE.py')

os.system('python ./training/train_gmm.py')