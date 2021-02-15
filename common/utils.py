import os
import pickle
import math
import time
import ast
import argparse
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.function_base import disp
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
    rows = math.ceil(len(disp_img_ids) / cols)
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
        
        ax.set_title("img# {} - Channel {}".format(disp_img_id, channel), fontsize=10, color = 'black')
        
        ax.set_xlabel("min: {:3d}  max: {:3d}  mean: {:.0f}".format(
                        X[disp_img_id][:,:,channel].min(), 
                        X[disp_img_id][:,:,channel].max(), 
                        X[disp_img_id][:,:,channel].mean()), fontsize=10, color = 'black')

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
    ax.set_xlabel("label: {:2d}-{:.25s}".format(y[disp_img_id], signnames[y[disp_img_id]]), fontsize=10, color = 'black')

    # if not grid:
    plt.grid(grid)
    surf = plt.imshow(X[disp_img_id].astype(np.uint8), cmap=cmap,
               norm=norm, interpolation=interpolation)
    # cbar = fig.colorbar(surf, shrink=0.65, aspect=30, fraction=0.05)
    i += 1

    # fig.tight_layout(rect=[0, 0.0, 1, 1], h_pad=h_pad, w_pad = w_pad)
    plt.show()    

##----------------------------------------------------------------------
## display_image
##----------------------------------------------------------------------    
def display_label_distribution(y, label_names, h_pad = 3, w_pad = 1, title = ' Training Dataset'):
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
    ax.set_title(title+' Label Distributions', fontsize=15)
    ax.set_xlabel(' Sign Class Id', fontsize = 12)    
    ax.set_ylabel(' Frequency', fontsize = 12)    
    ax.set_xlim([-0.5,42.5])
    for i, v in enumerate(y_series):
        ax.text(i- 0.25, v + 20, str(v), color='blue', fontweight='bold', fontsize=9)
    # ax.set_ylim(y_lims)    


    fig.tight_layout(rect=[0, 0.0, 1, 1], h_pad=h_pad, w_pad = w_pad)
    plt.show()                    
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


def display_inference_results(r_dict, verbose = False):

    # print(r_dict[k1].keys())

    print()
    print('                      Batch    Batches      Inference     ')
    print('  Key2      Epochs    Size     /Epoch       Accuracy      Weights File')
    for k2 in sorted(r_dict.keys(), key = lambda x: int(x.split(':')[1]) ):
        run_stats = r_dict[k2]
        best_val_acc = run_stats.get('best_val_acc', 0)
        print(' {:6s}   {:7d}  {:7d}   {:7d}       {:8.4f}       {}'.format(
                k2, run_stats['epochs'][-1],  run_stats['batch_size'],  run_stats['bpe'], run_stats['test_acc'][-1], run_stats['model_file']))
    return 


def display_training_results2(r_dict, k1, verbose = False):
    
    k1 = 'BS:'+str(k1)
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

def display_training_schedule(dropout = False,archs = [0,1,2,3,4,5,6,7,8],verbose = False):
    results_dir = './results/'
    tbl = {}

    for i in archs: 
        
        mdl = 'arch'+str(i)
        mdl_dropout = '_dropout' if dropout else ''
        tbl[mdl] = {}

        filename= mdl+mdl_dropout+'_results_test'
        pickle_file = results_dir + filename  
        results_dict = load_results(results_dir+filename)
       
        if results_dict != -1:
            load_key = list(results_dict.keys())[0]
            test_results = results_dict[load_key]
            results_keys = sorted(test_results.keys(), key = lambda x: int(x.split(':')[1]))
            print('-'*140)
            print('File : ', pickle_file, '   load_key : ',load_key,'   results_keys: ', results_keys)
            print('-'*140)
        else :
            print('File : ', pickle_file, ' not found')
            continue

        results_keys = sorted(test_results.keys(), key = lambda x: int(x.split(':')[1]))
        for bs_key in results_keys: 
            bs = int(bs_key.split(':')[-1])

            tbl[mdl][bs] = []

            run_stats = test_results[bs_key]
            print()
            print('{:6s}  Start     End     '.format(bs_key))
            print('        Epoch    Epoch     Learning Rate')
            print('        -----    ------    -------------')
            start_idx = 0
            curr_lr = run_stats['learning_rate'][start_idx]
            for idx, lr in enumerate(run_stats['learning_rate']):
                if lr != curr_lr :
                    print('      {:6d}    {:6d}     {:9.4f}    '.format(start_idx+1, idx, curr_lr))
                    curr_lr = lr
                    start_idx = idx
        
            print('      {:6d}    {:6d}     {:9.4f} '.format(start_idx+1, idx+1,curr_lr))
        print()
    return


def build_arch_training_results(dropout = False, archs = [0,1,2,3,4,5,6,7,8], verbose = False):
    results_dir = './results/'
    tbl = {}

    for i in archs: 
        
        mdl = 'arch'+str(i)
        mdl_dropout = '_dropout' if dropout else ''
        tbl[mdl] = {}
        
        filename= mdl+mdl_dropout+'_results_test'
        pickle_file = results_dir + filename  
        results_dict = load_results(results_dir+filename)
       
        if results_dict != -1:
            load_key = list(results_dict.keys())[0]
            test_results = results_dict[load_key]
            results_keys = sorted(test_results.keys(), key = lambda x: int(x.split(':')[1]))
            print('File : ', pickle_file, '   load_key : ',load_key,'   results_keys: ', results_keys)

        else :
            print('File : ', pickle_file, ' not found')
            continue

        
        for bs_key in results_keys:
            bs = int(bs_key.split(':')[-1])

            tbl[mdl][bs] = []

            run_stats = test_results[bs_key]

            at_epochs = [50,100,200,300,400,500]
            at_idx = 0
            if verbose:
                print()
                print(' {:6s}           Actual    Validation '.format(bs_key))
                print(' Ckpt   Epoch     Epoch     Accuracy      Checkpoint filename  ')
                print(' ----   -----    ------    ----------     ---------------------------------------')
            for idx, (ep,ck,ac) in enumerate(zip(run_stats['run_epochs'], run_stats['run_ckpts'],run_stats['run_ckpt_acc']),1):
                if ep >= at_epochs[at_idx]:
                    tbl[mdl][bs].append((ep, ac))
                    if verbose:
                        print(' {:4d}  {:6d}    {:6d}   {:9.4f}      {} '.format(idx,at_epochs[at_idx], ep,  ac,ck))
                    at_idx +=1
            if verbose:
                print(' {:4d}  {:6d}    {:6d}   {:9.4f}      {} '.format(idx,at_epochs[at_idx], ep,  ac,ck))
            tbl[mdl][bs].append((ep, ac))
        if verbose:
            print()
    return tbl


def display_arch_training_results(tbl, archs = [0,1,2,3,4,5,6,7,8], batch_size = None):
    if isinstance(batch_size, int):
        batch_size = [batch_size]
    elif batch_size is None:
        batch_size = [32, 64, 128, 256]
    print( ' table keys : ', tbl.keys())
    table_keys = list(tbl.keys())
    for  bs in batch_size:
        print('\nBatchsize: ', bs)
        hdr = '|'.join(['{:^10s}'.format(mdl) for mdl in table_keys])
        hdr = '| epochs |'+ hdr + '|'
        print(hdr)
        hdr = ':|:'.join(['--------'.format(mdl) for mdl in table_keys])
        hdr = '|:------:|:'+ hdr + ':|'
        print(hdr) 

        for ix, ep in enumerate([50,100,200,300,400,500]):
            ln = '|{:^8d}'.format(ep)
            for mdl in table_keys:
                try:
                    ln += '|{:^10.4f}'.format(tbl[mdl][bs][ix][1])  
                except IndexError:
                    # print(mdl,tbl[mdl][bs], ix)
                    ln += '|{:^10.4f}'.format(tbl[mdl][bs][-1][1])  
                except KeyError :
                    ln += '|  ------  '
            ln += '|'
            print(ln)
        
##----------------------------------------------------------------------
## plot training results
##----------------------------------------------------------------------   
def plot_training_results(r_dict, batch_size = None, batches = False, rolling_window = 1, title = ''):
    plt.style.use('ggplot')
 
    if batch_size is None:
        batch_size = sorted(r_dict.keys(), key = lambda x: int(x.split(':')[1]) )
    else: 
        batch_size = batch_size if isinstance(batch_size,list) else [batch_size]
    # print(' bs keys :', batch_size)

    for bs_key in batch_size:
        fig = plt.figure(figsize=(20, 5))
        
        units = 'batches' if batches else 'epochs'

        loss_plot = plt.subplot(121)
        loss_plot.set_title('{} - Training Results (Loss) -  batch size: {} '.format(title, r_dict[bs_key]['batch_size']))
        trn_rolling_mean = pd.Series(r_dict[bs_key]['trn_loss']).rolling(window=rolling_window).mean()
        val_rolling_mean = pd.Series(r_dict[bs_key]['val_loss']).rolling(window=rolling_window).mean()
        loss_plot.plot(r_dict[bs_key][units], trn_rolling_mean, 'orange', linestyle= 'dashed', label='Training Loss')
        loss_plot.plot(r_dict[bs_key][units], val_rolling_mean, 'green', label='Validation Loss')
        loss_plot.set_xlim([r_dict[bs_key][units][0]-1, r_dict[bs_key][units][-1]])
        loss_plot.set_ylim(-0.1, 2.5)
        loss_plot.set_xlabel(units)
        loss_plot.legend(loc='best')
        
        acc_plot = plt.subplot(122)
        acc_plot.set_title('{} - Training Results (Accuracy) - batch size: {}'.format(title, r_dict[bs_key]['batch_size']))
        trn_rolling_mean = pd.Series(r_dict[bs_key]['trn_acc']).rolling(window=rolling_window).mean()
        val_rolling_mean = pd.Series(r_dict[bs_key]['val_acc']).rolling(window=rolling_window).mean()
        acc_plot.plot(r_dict[bs_key][units],  trn_rolling_mean, 'orange', linestyle= 'dashed', label='Training Accuracy')
        acc_plot.plot(r_dict[bs_key][units],  val_rolling_mean, 'green' , label='Validation Accuracy')
        acc_plot.set_xlim([r_dict[bs_key][units][0]-1, r_dict[bs_key][units][-1]])
        acc_plot.set_ylim(-0.0, 1.05)
        acc_plot.set_xlabel(units)
        acc_plot.legend(loc='best')

        plt.tight_layout()
    plt.show()


def plot_training_results_by_batchsize(r_dict,  batches = False, rolling_window = 10, title = ''):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 10))
    key = ''
    batch_size = sorted(r_dict.keys(), key = lambda x: int(x.split(':')[1]) )
    print(' batch sizes:', batch_size)
    units = 'batches' if batches else 'epochs'
 
    y_lims = [0.6, 1.02]
    x_lims = [  0, max([r_dict[key][units][-1] for key in r_dict.keys()])]
    
    ##Training Loss by batch size 
    loss_plot = plt.subplot(221)
    loss_plot.set_title(title + ' - Training Loss by batch size')
    for key in batch_size:
        rolling_mean = pd.Series(r_dict[key]['trn_loss']).rolling(window=rolling_window).mean()
        loss_plot.plot(r_dict[key][units],  rolling_mean, label=key)
        # loss_plot.set_xlim([r_dict[key][units][0], r_dict[key][units][-1]])
    loss_plot.set_xlim(x_lims)
    loss_plot.set_xlabel(units)
    loss_plot.legend(loc='upper left')

    ##Training Accuracy by batch size 
    acc_plot = plt.subplot(222)
    acc_plot.set_title(title + ' - Training Accuracy by batch size')
    for key in batch_size:
        rolling_mean = pd.Series(r_dict[key]['trn_acc']).rolling(window=rolling_window).mean()
        acc_plot.plot(r_dict[key][units],  rolling_mean, label=key)
    acc_plot.set_xlabel(units)
    acc_plot.set_xlim(x_lims)
    acc_plot.set_ylim(y_lims)
    acc_plot.legend(loc='best')
    
    ##Validation Loss by batch size 
    loss_plot = plt.subplot(223)
    loss_plot.set_title(title + ' - Validation Loss by batch size')
    for key in batch_size:
        rolling_mean = pd.Series(r_dict[key]['val_loss']).rolling(window=rolling_window).mean()
        loss_plot.plot(r_dict[key][units],  rolling_mean, label=key)
    loss_plot.set_xlim(x_lims)
    loss_plot.set_xlabel(units)
    loss_plot.legend(loc='upper left')

    ##Training Accuracy by batch size 
    acc_plot = plt.subplot(224)
    acc_plot.set_title(title + '  - Validation accuracy by batch size')
    for key in batch_size:
        rolling_mean = pd.Series(r_dict[key]['val_acc']).rolling(window=rolling_window).mean()
        acc_plot.plot(r_dict[key][units],  rolling_mean, label=key)
    acc_plot.set_xlim(x_lims)
    acc_plot.set_ylim(y_lims)
    acc_plot.set_xlabel(units)
    acc_plot.legend(loc='best')

    plt.tight_layout()
    plt.show()



##----------------------------------------------------------------------
## Pickle file save and load
##----------------------------------------------------------------------   
def save_results2(pickle_file, results, results_key = None):
# Save the data for easy access
    if pickle_file.rfind('.pickle') == -1:
        pickle_file = pickle_file+'.pickle'

    print(' Saving data to pickle file...')
    if results_key is None:
        save_dict = results
    else:
        save_dict  = dict(results_key = results)
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                save_dict,
                pfile, 
                pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(' Unable to save data to', pickle_file, ':', e)
        raise
    else:
        print(' Data saved to pickle file.', pickle_file)
    return 1

def save_results(pickle_file, results, results_key):
# Save the data for easy access
    if pickle_file.rfind('.pickle') == -1:
        pickle_file = pickle_file+'.pickle'

    print(' Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {results_key: results},
                pfile, 
                pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(' Unable to save data to', pickle_file, ':', e)
        raise
    else:
        print(' Data saved to pickle file.', pickle_file)
    return 1

def load_results(pickle_file, verbose= False):
    if pickle_file.rfind('.pickle') == -1:
        pickle_file = pickle_file+'.pickle'

    # if os.path.isfile(pickle_file):
    try:
        # Reload the data
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except FileNotFoundError as e: 
        if verbose:
            print(' File not found : ', pickle_file)
        return -1
    except Exception as e:
        if verbose:
            print(' Unable to load data from', pickle_file, ':', e)
        raise
    else:
        if verbose:
            print(' Data loaded from pickle_data. load key : ', pickle_data.keys())
        return pickle_data    
    # else:
        # if verbose:
            # print(' Error pickle file not found ...')
        # return -1

##------------------------------------------------------------------------------------
## Parse command line arguments
##  
## Example:
## train-shapes_gpu --epochs 12 --steps-in-epoch 7 --last_epoch 1234 --logs_dir mrcnn_logs
## args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
##------------------------------------------------------------------------------------
def command_line_parser():
    def training_schedule(input):
        return eval(input)

    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')

    parser.add_argument('--model_config', 
                        required=True,
                        default='last',
                        metavar="model config selection",
                        help="MRCNN model weights file: 'coco' , 'init' , or Path to weights .h5 file ")

    parser.add_argument('--batch_sizes', 
                        required=False,
                        nargs = '+',
                        type  = int, 
                        # default=5,
                        metavar="<batch size>",
                        help="Number of data samples in each batch (default=5)")                    

    parser.add_argument('--dry-run', 
                        required=False,
                        default=False,
                        action='store_true',
                        help="dry run  ")

    parser.add_argument('--ckpt_dir', 
                        required=False,
                        default='./checkpoints/',
                        metavar="<checkpoint folder>",
                        help="Model checkpoints directory (default=logs/)")

    parser.add_argument('--results_filename', 
                        required=False,
                        default='_results_test',  
                        metavar="<results folder>",
                        help="Model checkpoints directory (default=./results/)")

    parser.add_argument('--results_dir', 
                        required=False,
                        default='./results/',
                        metavar="<results folder>",
                        help="Model checkpoints directory (default=./results/)")

    parser.add_argument('--training_schedule', 
                        required=True,
                        nargs = '+',
                        default='(100, 0.001)', type=training_schedule, 
                        metavar="<active coco classes>",
                        help="<identifies active coco classes" )

    parser.add_argument('--epochs', required=False,
                        default=1,
                        metavar="<epochs to run>",
                        help="Number of epochs to run (default=3)")
                        
    parser.add_argument('--steps_in_epoch', required=False,
                        default=1,
                        metavar="<steps in each epoch>",
                        help="Number of batches to run in each epochs (default=5)")

    parser.add_argument('--val_steps', required=False,
                        default=1,
                        metavar="<val steps in each epoch>",
                        help="Number of validation batches to run at end of each epoch (default=1)")
                        
    # parser.add_argument("command",
                        # metavar="<command>",
                        # help="'train' or 'evaluate' on MS COCO")

    # parser.add_argument('--dataset', required=True,
                        # metavar="/path/to/coco/",
                        # help='Directory of the MS-COCO dataset')
    
    # parser.add_argument('--limit', required=False,
                        # default=500,
                        # metavar="<image count>",
                        # help='Images to use for evaluation (defaults=500)')

    # parser.add_argument('--mrcnn_exclude_layers', 
    #                     required=False,
    #                     nargs = '+',
    #                     type=str.lower, 
    #                     metavar="/path/to/weights.h5",
    #                     help="layers to exclude from loading from weight file" )
                        

    # parser.add_argument('--mrcnn_layers', 
    #                     required=False,
    #                     nargs = '+',
    #                     default=['mrcnn', 'fpn', 'rpn'], type=str.lower, 
    #                     metavar="/path/to/weights.h5",
    #                     help="MRCNN layers to train" )
                        
    # parser.add_argument('--evaluate_method', 
    #                     required=False,
    #                     choices = [1,2,3],
    #                     default=1, type = int, 
    #                     metavar="<evaluation method>",
    #                     help="Detection Evaluation method : [1,2,3]")
                        
    # parser.add_argument('--fcn_model', 
    #                     required=False,
    #                     default='last',
    #                     metavar="/path/to/weights.h5",
    #                     help="FCN model weights file: 'init' , or Path to weights .h5 file ")

    # parser.add_argument('--fcn_logs_dir', required=False,
    #                     default='train_fcn',
    #                     metavar="/path/to/logs/",
    #                     help="FCN Logs and checkpoints directory (default=logs/)")

    # parser.add_argument('--fcn_arch', required=False,
    #                     choices=['FCN32', 'FCN16', 'FCN8', 'FCN8L2', 'FCN32L2'],
    #                     default='FCN32', type=str.upper, 
    #                     metavar="/path/to/weights.h5",
    #                     help="FCN Architecture : fcn32, fcn16, or fcn8")

    # parser.add_argument('--fcn_layers', required=False,
    #                     nargs = '+',
    #                     default=['fcn32+'], type=str.lower, 
    #                     metavar="/path/to/weights.h5",
    #                     help="FCN layers to train" )

    # parser.add_argument('--fcn_losses', required=False,
    #                     nargs = '+',
    #                     default='fcn_BCE_loss', 
    #                     metavar="/path/to/weights.h5",
    #                     help="FCN Losses: fcn_CE_loss, fcn_BCE_loss, fcn_MSE_loss" )
                        
    # parser.add_argument('--fcn_bce_loss_method', required=False,
    #                     choices = [1,2],
    #                     default=1, type = int, 
    #                     metavar="<BCE Loss evaluation method>",
    #                     help="Evaluation method : [1: Loss on all classes ,2: Loss on one class only]")
                        
    # parser.add_argument('--fcn_bce_loss_class', required=False,
    #                     default=0, type = int, 
    #                     metavar="<BCE Loss evaluation class>",
    #                     help="Evaluation class")

    # parser.add_argument('--last_epoch', required=False,
    #                     default=0,
    #                     metavar="<last epoch ran>",
    #                     help="Identify last completed epcoh for tensorboard continuation")
                        

    # parser.add_argument('--scale_factor', required=False,
    #                     default=4,
    #                     metavar="<heatmap scale>",
    #                     help="Heatmap scale factor")                    

    # parser.add_argument('--lr', required=False,
    #                     default=0.001,
    #                     metavar="<learning rate>",
    #                     help="Learning Rate (default=0.001)")

    # parser.add_argument('--opt', required=False,
    #                     default='adagrad', type = str.upper,
    #                     metavar="<optimizer>",
    #                     help="Optimization Method: SGD, RMSPROP, ADAGRAD, ...")
                        
    # parser.add_argument('--sysout', required=False,
    #                     choices=['SCREEN', 'HEADER', 'ALL'],
    #                     default='screen', type=str.upper,
    #                     metavar="<sysout>",
    #                     help="sysout destination: 'screen', 'header' , 'all' (header == file) ")

    # parser.add_argument('--new_log_folder', required=False,
    #                     default=False, action='store_true',
    #                     help="put logging/weights files in new folder: True or False")

    # parser.add_argument('--dataset', required=False,
    #                     choices=['newshapes', 'newshapes2', 'coco2014'],
    #                     default='newshapes', type=str, 
    #                     metavar="<Toy dataset type>",
    #                     help="<identifies toy dataset: newshapes or newshapes2" )
    
    return parser

        
def display_input_parms(args):
    """Display Configuration values."""
    print("\n   Arguments passed :")
    print("   --------------------")
    for a in dir(args):
        if not a.startswith("__") and not callable(getattr(args, a)):
            print("   {:30} {}".format(a, getattr(args, a)))
    print("\n")
 
def get_from_config(config_filename, arch):
    config_dict = {arch : {}}

    config_file = configparser.ConfigParser()
    config_file.read(config_filename)
    print(' Model configurations: ', config_file.sections())

    for opt in config_file[arch].keys():
        # print(opt, '    ', config_file[arch][opt])
        if opt == 'ckpt_prefix':
            config_dict[arch][opt] = config_file.get(arch, opt)
        elif opt in ['drop_rate', 'sigma', 'mu']:
            config_dict[arch][opt] = config_file.getfloat(arch,opt)
        else:
            config_dict[arch][opt] = config_file.getint(arch,opt)
    
    config_dict[arch]['ckpt_prefix'] = arch
    return config_dict
