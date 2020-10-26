import os
import pickle
import math
import time
import numpy as np
from numpy.lib.function_base import disp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import defaultdict

def results_dict():
    return defaultdict(results_dict)

    
##----------------------------------------------------------------------
## display_image
##----------------------------------------------------------------------      
def preprocess_data(X, y, n_classes):
    X_scaled = ((X - [128]) / 128).astype(np.float32)

    print(' Before Scaling : X max: ',  X.max(),  '  min: ', X.min(), ' X data type: ', X.dtype)
    print('  After Scaling : X  max: ',  X_scaled.max(),  '  min: ',X_scaled.min(), ' shape: ', X_scaled.shape , ' data type: ', X_scaled.dtype)

    return X_scaled, y 


##----------------------------------------------------------------------
## display_random_images
##----------------------------------------------------------------------
def display_random_images(X,y, signnames, shape=(2,8), cmap=None, norm=None,
                   interpolation=None, width=25, grid = False, 
                   ticks = False, h_pad = 3, w_pad = 1):
    """
    Display the given set of images, optionally with titles.
    
    images:             list or array of image tensors in HWC format.
    titles:             optional. A list of titles to display with each image.
    cols:               number of images per row
    cmap:               Optional. Color map to use. For example, "Blues".
    norm:               Optional. A Normalize instance to map values to colors.
    interpolation:      Optional. Image interporlation to use for display.
    """
    
#     titles = titles if titles is not None else [None] * len(images)
    rows = shape[0]
    cols = shape[1]
    disp_img_ids = np.random.randint(0, X.shape[0], (rows,cols))
    disp_img_ids

    
    #    print('titles is :', titles) 
    fig = plt.figure(figsize=(2.5*cols, 3 * rows))
    
    i = 1
    for row in disp_img_ids:
        for id in row:
            ax = plt.subplot(rows, cols, i)
            if ticks:
                ax.tick_params(axis='both', bottom = True, left = True, labelsize = 6, width = 1.0, length = 2)
            else:
                ax.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)

            ax.set_title("img# {}".format(id), fontsize=9)
            ax.set_xlabel("label: {:2d}-{:.25s}".format(y[id], signnames[y[id]][:10]), fontsize=9)

            # if not grid:
            plt.grid(grid)
            plt.imshow(X[id].astype(np.uint8), cmap=cmap,
                       norm=norm, interpolation=interpolation)
            i += 1

    fig.tight_layout(rect=[0, 0.0, 1, 1], h_pad=h_pad, w_pad = w_pad)
    plt.show()

##----------------------------------------------------------------------
## display_images
##----------------------------------------------------------------------
def display_images(X,y, signnames, disp_img_ids = None , shape=(4,8), cmap=None, norm=None,
                   interpolation=None, width=25, grid = False, 
                   ticks = False, h_pad = 3, w_pad = 1):
    """
    Display the given set of images, optionally with titles.
    
    images:             list or array of image tensors in HWC format.
    titles:             optional. A list of titles to display with each image.
    cols:               number of images per row
    cmap:               Optional. Color map to use. For example, "Blues".
    norm:               Optional. A Normalize instance to map values to colors.
    interpolation:      Optional. Image interporlation to use for display.
    """
    if disp_img_ids is None:
        disp_img_ids = np.arange(X.shape[0])

    cols = min(shape[1], len(disp_img_ids))
    rows = min(shape[0], math.ceil(len(disp_img_ids)// cols))
    
    fig = plt.figure(figsize=(2.5*cols, 3 * rows))
    
    i = 1
    for id in disp_img_ids:
        ax = plt.subplot(rows, cols, i)
        if ticks:
            ax.tick_params(axis='both', bottom = True, left = True, labelsize = 6, width = 1.0, length = 2)
        else:
            ax.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)

        ax.set_title("img# {}".format(id), fontsize=9)
        ax.set_xlabel("label: {:2d}-{:.25s}".format(y[id], signnames[y[id]]), fontsize=9)

        # if not grid:
        plt.grid(grid)
        plt.imshow(X[id].astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1

    fig.tight_layout(rect=[0, 0.0, 1, 1], h_pad=h_pad, w_pad = w_pad)
    plt.show()

##----------------------------------------------------------------------
## display a single image 
##----------------------------------------------------------------------
def display_image(X,y, signnames, disp_img_id , cmap=None, norm=None,
                   interpolation=None, width=25, grid = False, 
                   ticks = False, h_pad = 3, w_pad = 1):
    """
    Display the given set of images, optionally with titles.
    
    images:             list or array of image tensors in HWC format.
    titles:             optional. A list of titles to display with each image.
    cols:               number of images per row
    cmap:               Optional. Color map to use. For example, "Blues".
    norm:               Optional. A Normalize instance to map values to colors.
    interpolation:      Optional. Image interporlation to use for display.
    """
    rows = 1
    cols = 5
    fig = plt.figure(figsize=(4*cols, 5 * rows))
    
    i = 1
    for channel in [0,1,2]:
        ax = plt.subplot(rows, cols, i)
        if ticks:
            ax.tick_params(axis='both', bottom = True, left = True, labelsize = 6, width = 1.0, length = 2)
        else:
            ax.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)
        
        ax.set_title("img# {} - Channel {}".format(disp_img_id, channel), fontsize=9)
        
        ax.set_xlabel("min: {:3d}  max: {:3d}  mean: {:.0f}".format(
                        X[disp_img_id][:,:,channel].min(), 
                        X[disp_img_id][:,:,channel].max(), 
                        X[disp_img_id][:,:,channel].mean()), fontsize=9)

        # if not grid:
        plt.grid(grid)
        surf = plt.imshow(X[disp_img_id][:,:,channel], cmap=cmap,
                   norm=norm, interpolation=interpolation)
        cbar = fig.colorbar(surf, shrink=0.7, aspect=30, fraction=0.05)
        i += 1

    ax = plt.subplot(rows, cols, i)
    if ticks:
        ax.tick_params(axis='both', bottom = True, left = True, labelsize = 6, width = 1.0, length = 2)
    else:
        ax.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)

    ax.set_title("img# {}".format(disp_img_id), fontsize=9)
    ax.set_xlabel("label: {:2d}-{:.25s}".format(y[disp_img_id], signnames[y[disp_img_id]]), fontsize=9)

    # if not grid:
    plt.grid(grid)
    surf = plt.imshow(X[disp_img_id].astype(np.uint8), cmap=cmap,
               norm=norm, interpolation=interpolation)
    cbar = fig.colorbar(surf, shrink=0.7, aspect=30, fraction=0.05)
    i += 1

    fig.tight_layout(rect=[0, 0.0, 1, 1], h_pad=h_pad, w_pad = w_pad)
    plt.show()    

##----------------------------------------------------------------------
## display_image
##----------------------------------------------------------------------    
def display_label_distribution(y, label_names):
    y_series = pd.Series(y).value_counts().sort_index()
#     y_series = pd.Series(y_train)
#     y_series.value_counts()    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(25,6))
    x = np.arange(0,43,1)
#     y1 = np.random.randint(100, 2500, size=(43))
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    width = 0.6
    ax.bar(x, y_series, width)
    # ax.bar(x + width, y2, width,
    #         color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])
    plt.xticks(rotation=30)
    ax.set_xticks(x + width//2)
    ax.set_xticklabels( np.arange(0,43,1))
    ax.set_title(' Sign Distributions', fontsize=15)
    ax.set_xlabel(' Sign label id', fontsize = 12)    
    ax.set_ylabel(' Frequency', fontsize = 12)    
    

                
##----------------------------------------------------------------------
## display_image
##----------------------------------------------------------------------   
def display_training_results(r_dict, verbose = False):

    # print(r_dict[k1].keys())

    print()
    print('                      Batch    Batches             Best          Training                           Validation ')
    print('  Key2      Epochs    Size     /Epochs     LR      VAcc     Acc[St]     Acc[End]    Loss[St]   Loss[End]    Acc[St]     Acc[End]')
    for k2 in sorted(r_dict.keys(), key = lambda x: int(x.split(':')[1]) ):
        run_stats = r_dict[k2]
        best_val_acc = run_stats.get('best_val_acc', 0)
        print(' {:6s}   {:7d}  {:7d}   {:7d}   {:7.5f}   {:7.4f}   {:8.4f}    {:8.4f}    {:8.4f}    {:8.4f}    {:8.4f}    {:8.4f}  '.format(
                k2, run_stats['epochs'][-1],  run_stats['batch_size'],  run_stats['bpe'],  run_stats['lr'] , best_val_acc ,
                run_stats['trn_acc'][1],  run_stats['trn_acc'][-1], 
                run_stats['val_loss'][1], run_stats['val_loss'][-1],
                 run_stats['val_acc'][1], run_stats['val_acc'][-1]))
    return 


def display_training_results2(r_dict, k1, verbose = False):
    
    print('im here')
    print(r_dict[k1].keys())

    print()
    print('                      Batch    Batches             Best          Training                           Validation ')
    print('  Key2      Epochs    Size     /Epochs     LR      VAcc     Acc[St]     Acc[End]    Loss[St]   Loss[End]    Acc[St]     Acc[End]')
    run_stats = r_dict[k1]
    best_val_acc = run_stats.get('best_val_acc', 0)
    print(' {:6s}   {:7d}  {:7d}   {:7d}   {:7.5f}   {:7.4f}   {:8.4f}    {:8.4f}    {:8.4f}    {:8.4f}    {:8.4f}    {:8.4f}  '.format(
            k1, run_stats['epochs'][-1],  run_stats['batch_size'],  run_stats['bpe'],  run_stats['lr'] , best_val_acc ,
            run_stats['trn_acc'][1],  run_stats['trn_acc'][-1], 
            run_stats['val_loss'][1], run_stats['val_loss'][-1],
            run_stats['val_acc'][1], run_stats['val_acc'][-1]))

    print()
    print('                  Ckpt Val ')
    print(' Ckpt   Epochs    Accuracy      Checkpoint filename  ')
    print(' ----   ------    --------      ---------------------------------------')
    for idx, (ep,ck,ac) in enumerate(zip(run_stats['run_epochs'], run_stats['run_ckpts'],run_stats['run_ckpt_acc']),1):
        print(' {:4d}  {:6d}     {:7.4f}      {} '.format(idx, ep, ac,ck))

    # print('                                                             ')
    # print('   Key    Epochs   BatchSize  Batches/Epoch    LR Epoch    Trn Acc[0]   Trn Acc[End]     Vld Acc[St]     Val Acc[End]')
    # print(' {:5}   {:5d}    {:10d}    {:10d}    {:10.4f}    {:10.4f}    {:10.4f}    {:10.4f}    {:10.4f}    '.format(
            # k1, r_dict[k1]['epochs'][-1],  r_dict[k1]['batch_size'],  r_dict[k1]['bpe'],  r_dict[k1]['lr'] ,
            # r_dict[k1]['trn_acc'][0], r_dict[k1]['trn_acc'][-1], r_dict[k1]['val_acc'][0], r_dict[k1]['val_acc'][-1]))
    if verbose:
        print()
        print('   Epoch      Batches       LR       Trn Acc     Trn Loss      Vld Acc     Val Loss')
        print('   -----      -------     ------     -------     --------      -------     --------')
        for e, b, lr, ta, tl, va, vl  in zip(r_dict[k1]['epochs'] , r_dict[k1]['batches'], r_dict[k1]['learning_rate'], 
                                         r_dict[k1]['trn_acc'], r_dict[k1]['trn_loss'],
                                         r_dict[k1]['val_acc'], r_dict[k1]['val_loss']):
            print('   {:5d}   {:10d}  {:10.4f}  {:10.4f}   {:10.4f}   {:10.4f}   {:10.4f}'.format(e,b,lr,ta,tl, va,vl))
    return 

##----------------------------------------------------------------------
## plot training results
##----------------------------------------------------------------------   
def plot_training_results(r_dict, bs_keys = None, batches = False, rolling_window = 1):
    plt.style.use('ggplot')
    print(r_dict.keys())
    if bs_keys is None:
        bs_keys = sorted(r_dict.keys(), key = lambda x: int(x.split(':')[1]) )
    else: 
        bs_keys = bs_keys if isinstance(bs_keys,list) else [bs_keys]
    print(' bs keys :', bs_keys)

    for bs_key in bs_keys:
        fig = plt.figure(figsize=(20, 5))
        
        units = 'batches' if batches else 'epochs'

        loss_plot = plt.subplot(121)
        loss_plot.set_title('Loss : batch size:  {} '.format(r_dict[bs_key]['batch_size']))
        trn_rolling_mean = pd.Series(r_dict[bs_key]['trn_loss']).rolling(window=rolling_window).mean()
        val_rolling_mean = pd.Series(r_dict[bs_key]['val_loss']).rolling(window=rolling_window).mean()
        loss_plot.plot(r_dict[bs_key][units], trn_rolling_mean, 'orange', linestyle= 'dashed', label='Training Loss')
        loss_plot.plot(r_dict[bs_key][units], val_rolling_mean, 'g', label='Validation Loss')
        loss_plot.set_xlim([r_dict[bs_key][units][0], r_dict[bs_key][units][-1]])
        loss_plot.set_xlabel(units)
        loss_plot.legend(loc='best')
        
        acc_plot = plt.subplot(122)
        acc_plot.set_title('Accuracy : batch size:  {}'.format(r_dict[bs_key]['batch_size']))
        trn_rolling_mean = pd.Series(r_dict[bs_key]['trn_acc']).rolling(window=rolling_window).mean()
        val_rolling_mean = pd.Series(r_dict[bs_key]['val_acc']).rolling(window=rolling_window).mean()
        acc_plot.plot(r_dict[bs_key][units],  trn_rolling_mean, 'orange', linestyle= 'dashed', label='Training Accuracy')
        acc_plot.plot(r_dict[bs_key][units],  val_rolling_mean, 'r' , label='Validation Accuracy')
        acc_plot.set_xlim([r_dict[bs_key][units][0], r_dict[bs_key][units][-1]])
        acc_plot.set_xlabel(units)
        acc_plot.legend(loc='best')

        plt.tight_layout()
    plt.show()


def plot_training_results_by_batchsize(r_dict,  batches = False, rolling_window = 10):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 10))
    key = ''
    bs_keys = sorted(r_dict.keys(), key = lambda x: int(x.split(':')[1]) )
    units = 'batches' if batches else 'epochs'
 
    y_lims = [0.6, 1.02]
    x_lims = [  0, max([r_dict[key][units][-1] for key in r_dict.keys()])]
    
    ##Training Loss by batch size 
    loss_plot = plt.subplot(221)
    loss_plot.set_title('Training Loss by batch size')
    for key in bs_keys:
        rolling_mean = pd.Series(r_dict[key]['trn_loss']).rolling(window=rolling_window).mean()
        loss_plot.plot(r_dict[key][units],  rolling_mean, label=key)
        # loss_plot.set_xlim([r_dict[key][units][0], r_dict[key][units][-1]])
    loss_plot.set_xlim(x_lims)
    loss_plot.set_xlabel(units)
    loss_plot.legend(loc='upper left')
    
    ##Validation Loss by batch size 
    loss_plot = plt.subplot(222)
    loss_plot.set_title('Validation Loss by batch size')
    for key in bs_keys:
        rolling_mean = pd.Series(r_dict[key]['val_loss']).rolling(window=rolling_window).mean()
        loss_plot.plot(r_dict[key][units],  rolling_mean, label=key)
    loss_plot.set_xlim(x_lims)
    loss_plot.set_xlabel(units)
    loss_plot.legend(loc='upper left')

    ##Training Accuracy by batch size 
    acc_plot = plt.subplot(223)
    acc_plot.set_title('Training accuracy by batch size')
    for key in bs_keys:
        rolling_mean = pd.Series(r_dict[key]['trn_acc']).rolling(window=rolling_window).mean()
        acc_plot.plot(r_dict[key][units],  rolling_mean, label=key)
    acc_plot.set_xlabel(units)
    acc_plot.set_xlim(x_lims)
    acc_plot.set_ylim(y_lims)
    acc_plot.legend(loc='best')

    ##Training Accuracy by batch size 
    acc_plot = plt.subplot(224)
    acc_plot.set_title('Validation accuracy by batch size')
    for key in bs_keys:
        rolling_mean = pd.Series(r_dict[key]['val_acc']).rolling(window=rolling_window).mean()
        acc_plot.plot(r_dict[key][units],  rolling_mean, label=key)
    acc_plot.set_xlim(x_lims)
    acc_plot.set_ylim(y_lims)
    acc_plot.set_xlabel(units)
    acc_plot.legend(loc='best')

    plt.tight_layout()
    plt.show()



##----------------------------------------------------------------------
## plot results
##----------------------------------------------------------------------   

def save_results(filename, results, results_name):
# Save the data for easy access
    pickle_file = filename+'.pickle'

    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {results_name: results},
                pfile, 
                pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    else:
        print('Data cached in pickle file.', pickle_file)
    return 

def load_results(filename):
    pickle_file = filename+'.pickle'

    if os.path.isfile(pickle_file):
        try:
            # Reload the data
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except Exception as e:
            print('Unable to load data from', pickle_file, ':', e)
            raise
    else:
        print('Error pickle file not found ...')

    print('Data loaded from pickle_data. Variable name : ', pickle_data.keys())
    return pickle_data    
