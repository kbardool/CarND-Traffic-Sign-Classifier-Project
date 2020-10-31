# Load pickled data
import os, sys
import pickle
import math
import time
import configparser
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from datetime import datetime, time

sys.path.append('.')
# print(sys.path)

# TF_CPP_MIN_LOG_LEVEL
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow.keras.preprocessing as kp
from tensorflow.python.client import device_lib
# tf.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.disable_v2_behavior()

# from common.utils import *
from common.utils import command_line_parser, display_input_parms, get_from_config,save_results,load_results
from common.model import run_model_training 


print(' Tensorflow Version      : ', tf.__version__)
print(' Built with CUDA support : ', tf.test.is_built_with_cuda())
print(' GPU available           : ', tf.test.is_gpu_available())
print(' List of GPU Devices     : ', tf.config.list_physical_devices('GPU'))
print(' TF Logging verbosity    : ', tf.logging.get_verbosity())
# print(device_list)
# print(type(device_list), len(device_list), type(device_list[0]))
print()
print('Device Name           Type           Memory Limit    Description')
print('-----------------     -------        ------------    ---------------')
for i,dev in enumerate(device_lib.list_local_devices()):
    print('{:20s}  {:10s}  {:15d}    {}'.format(dev.name, dev.device_type, dev.memory_limit, dev.physical_device_desc))
print()

pp = pprint.PrettyPrinter(indent=2, width=130)
# suffix = datetime.now().strftime("%y%m%d%H%M%S")
# print(suffix)

start_time = datetime.now()
# start_time_disp = start_time.strftime("%m-%d-%Y @ %H:%M:%S")
print('\n --> Execution started at:', start_time)


##------------------------------------------------------------------------------------
## Parse command line arguments
##------------------------------------------------------------------------------------
parser = command_line_parser()
args = parser.parse_args()
display_input_parms(args)

## Read Model cfg  

config_file = configparser.ConfigParser()
config_file.read('model_config.cfg')
print(' Model configurations: ', config_file.sections())

MODEL_ARCH = args.model_config
mdl_config = get_from_config(config_file, MODEL_ARCH)
# pp.pprint(mdl_config[MODEL_ARCH])


##------------------------------------------------------------------------------------
## Read data files
##------------------------------------------------------------------------------------
# del X_train, y_train, X_valid, y_valid
training_file = 'F:/MLDatasets/traffic-signs-data/train.p'
validation_file= 'F:/MLDatasets/traffic-signs-data/valid.p'
testing_file = 'F:/MLDatasets/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test   = test['features'], test['labels']

# Number of training exampLles
n_train = train['features'].shape[0]
n_validation = valid['features'].shape[0]
n_test = test['labels'].shape[0]
image_shape = train['features'].shape[1:]

# How many unique classes/labels there are in the dataset.
y_df = pd.DataFrame(train['labels'])
y_ids = sorted(pd.unique(y_df[0]).tolist())
print(' Number of unique labels : {} \n Labels: {} '.format(len(y_ids), y_ids))

n_classes = len(y_ids)
print()
print(" Number of training examples ....... ", n_train)
print(" Number of validation examples ..... ", n_validation)
print(" Number of testing examples ........ ", n_test)
print(" Image data shape .................. ", image_shape)
print(" Number of classes ................. ", n_classes)

signnames_csv = pd.read_csv('signnames.csv')
signnames = signnames_csv["SignName"].str.strip().to_dict()
sign_keys = signnames.keys()
sign_len = len(sign_keys)
half_len = math.ceil( sign_len / 2 )

print()
print('  Id    Sign description               \t\t\t\t  Id    Sign description               ')
print(' ----   -------------------------------\t\t\t\t ----   -------------------------------')


for i in range(0,half_len,1):
    print('  {:2d}    {:45s}'.format(i,signnames[i]), end = '')
    if (i+half_len) < sign_len:
        print('\t\t  {:2d}    {:45s}'.format(i+half_len,signnames[i+half_len]))
    else:
        print('\t\t ','-'*20)

# Remove the previous weights and bias
simple_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization = True,
                     data_format = 'channels_last')

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization = True,
                     rotation_range=10,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     vertical_flip = False,
                     data_format = 'channels_last')

print('\n')
print(' Datagen parameters: ')
print(' ------------------- ')
for i,v in data_gen_args.items():
    print('    {:.<35s} {}'.format(i,v) )

trn_datagen = kp.image.ImageDataGenerator( **data_gen_args)
vld_datagen = kp.image.ImageDataGenerator( **data_gen_args)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

# trn_datagen = kp.image.ImageDataGenerator( **simple_gen_args)
# vld_datagen = kp.image.ImageDataGenerator( **simple_gen_args)

trn_datagen.fit(X_train)
vld_datagen.fit(X_valid)

num_trn_examples = len(X_train)
num_val_examples = len(X_valid)

print(X_train.shape, X_valid.shape)

##-------------------------------------------------------------------------
##
##-------------------------------------------------------------------------
MODEL_ARCH = args.model_config

print('\n\n')

if args.results_filename is None:
    results_filename = MODEL_ARCH+'_results_new'
    results_key      = MODEL_ARCH+'_results_new'
    print(' Build NEW results file ', args.results_dir + results_filename )
    if os.path.isfile(args.results_dir + results_filename + '.pickle'):
        raise FileExistsError(' File '+ args.results_dir + results_filename + '.pickle', ' Already exists !!')
    results = defaultdict(defaultdict)
else:
    results_filename = args.results_filename
    if os.path.isfile(args.results_dir + results_filename + '.pickle'):
        results_dict = load_results(args.results_dir + args.results_filename)
        results_key  = list(results_dict.keys())[0]
        print(' Load existing results file ', args.results_dir + results_filename)
        results = results_dict[results_key]
    else:
        print(' Build new results file ', args.results_dir + results_filename)
        results_key = args.results_filename
        results = defaultdict(defaultdict)
        print(results.keys())

    print('      Results key is ', results_key)
    
BATCH_SIZES = args.batch_sizes     ## [256, 128, 64, 32, 16]
TRAINING_SCHEDULE = [(200, 0.001), (100, 0.0005), (100, 0.0002), (100, 0.0001)]

# TRAINING_SCHEDULE = [(2, 0.001), (2, 0.0005), (2, 0.0002), (2, 0.0001)]
# TRAINING_SCHEDULE = [(100, 0.0002), (100, 0.0001)]

# reload = True
# BS_KEY = 'BS:'+str(BATCH_SIZES [0])

##----------------------------------------------------------------------
print()
print(' Model parameters: ')
print(' ------------------- ')
for i,v in mdl_config[MODEL_ARCH].items():
    print('    {:.<35s} {}'.format(i,v) )

# print(' Checkpt prefix  : ' , mdl_config[MODEL_ARCH]['ckpt_prefix'])
# print(' Results Filename: ' , results_filename, '\t\t Results Dictionary Keyname:', results_dict)
# print(' Reload flag     : ' , reload)
# try:
    # print(' last model ckpt: ', results[BS_KEY]['last_ckpt'])
    # print(' last epoch ran : ', results[BS_KEY]['epochs'][-1])
# except:
    # print(' reload flag set to FALSE')
    # reload = False
    
# print(' Reload flag    : ', reload)
# print(' Batch size key : ', BS_KEY)


for BATCH_SIZE in BATCH_SIZES:

    BS_KEY = 'BS:'+str(BATCH_SIZE)

    reload = True
    print()
    print(' Batchsize ', BATCH_SIZE, ' run parameters:')
    print(' ---------------------------------------')
    print('    Model Config       : ', MODEL_ARCH)
    print('    Checkpoint prefix  : ', mdl_config[MODEL_ARCH]['ckpt_prefix'])
    print('    Results filename   : ', results_filename, '\t\t Results Dictionary Keyname:', results_key)
    print('    Training schedule  : ', TRAINING_SCHEDULE)
    print('    Reload flag        : ', reload)
    try:
        print('    Last model ckpt   : ', results[BS_KEY]['last_ckpt'])
        print('    Last epoch ran    : ', results[BS_KEY]['epochs'][-1])
    except:
        print('    Reload flag set to FALSE')
        reload = False
    print('    Reload flag        : ', reload)
    print('    Batch size key     : ', BS_KEY)
    print('\n')

    trn_datagen_flow = trn_datagen.flow(X_train, y_train, batch_size= BATCH_SIZE)
    vld_datagen_flow = vld_datagen.flow(X_valid, y_valid, batch_size= BATCH_SIZE)

    trn_bpe = math.ceil(num_trn_examples/ BATCH_SIZE)
    val_bpe = math.ceil(num_val_examples/ BATCH_SIZE)

    for EP,LR in TRAINING_SCHEDULE[0:]:
        
        reload = True
        print('\n Batch size: ',  BATCH_SIZE, ' Learning Rate: ', LR, ' # Epochs: ', EP)
        
        if reload:
            try:
                load_ckpt = results[BS_KEY]['last_ckpt']
                print(' checkpoint to load: ', load_ckpt)
            except:
                print(' Reload flag set to FALSE')
                reload = False
                load_ckpt = None
                results[BS_KEY] = defaultdict(list)
        
        if args.dry_run:
            print('------------ DRY RUN ------------')
            continue

        run_model_training( trn_datagen_flow,
                            vld_datagen_flow, 
                            mdl_config[MODEL_ARCH],
                            results = results[BS_KEY],
                            epochs = EP, 
                            learning_rate = LR, 
                            batch_size = BATCH_SIZE, 
                            training_batches = trn_bpe, 
                            validation_batches = val_bpe, 
                            reload = reload,
                            ckpt_file = load_ckpt)

        save_results(args.results_dir + results_filename, results, results_key )

        
##----------------------------------------------------------------------------------------------
## If in debug mode write stdout intercepted IO to output file
##----------------------------------------------------------------------------------------------            
end_time = datetime.now()     ## .strftime("%m-%d-%Y @ %H:%M:%S")
# if args.sysout in  ['ALL']:
    # print(' --> Execution ended at:', end_time)
    # sys.stdout.flush()
    # f_obj.close()    
    # sys.stdout = sys.__stdout__
    # print(' Run information written to ', sysout_name)    
print('\n Execution time :', end_time - start_time)
print('\n --> Execution ended at:',end_time)
exit(' Execution terminated ' ) 
            
