from collections import defaultdict
from common.utils import results_dict
import math
import tensorflow.compat.v1 as tf
from tqdm import tqdm, trange
from sklearn.utils import shuffle
from datetime import datetime

def run_model_training(trn_datagen, vld_datagen, model_config, results = None, 
                       starting_epoch = None , epochs = 1, batch_size = 64, learning_rate = 0.001,
                       training_batches = None, 
                       validation_batches = None,
                       reload = False , 
                       ckpt_file = None):

    tf.reset_default_graph()
    x_input = tf.placeholder(tf.float32, (None, 32, 32, 3), name = 'x_input')
    y_input = tf.placeholder(tf.int32, (None), name = 'y_input')
    def_weights_biases(model_config)   
    logits = model(x_input, model_config)

    y_onehot = tf.one_hot(y_input, 43)
    loss_operation, training_operation = model_optimization(logits, y_onehot, LR = learning_rate)
    accuracy_operation = model_accuracy(logits, y_onehot)

    if starting_epoch is None :
        starting_epoch = results['epochs'][-1] + 1 if reload else 1

    if results is None:
        results = defaultdict(list)
    
    if reload:
        pass
    else:
        results['best_val_acc'] = 0.8

    results['lr'] = learning_rate
    results['bpe'] = training_batches
    results['val_bpe'] = validation_batches
    results['batch_size'] = batch_size
    results['drop_rate']= model_config['drop_rate']

    logging_interval = 50
    ending_epoch = starting_epoch + epochs - 1
    epoch_i = 0
    loss = 0   
    acc  = 0
    ttl_batches  = 0 
    val_accuracy = 0.0
    val_loss = 9999.9
    trn_accuracy = 0.0
    trn_loss = 9999.9
    
    print('\n--- Current run parameters ---------------------------------------------------------------')
    print(' Training Epochs          : ', epochs, '   Last Epoch Ran :', starting_epoch -1,
          '\n Starting epoch           : ', starting_epoch, '    Ending epoch :', ending_epoch,
          '\n Batch Size               : ', batch_size, 
          '\n Learning rate            : ', learning_rate, 
          '\n Dropout rate             : ', model_config['drop_rate'],
          '\n Training Batches/epoch   : ', training_batches, 
          '\n Validation Batches/epoch : ', validation_batches,
          '\n Validation Acc Ckpt Thrshld : ', '{:7.4f}'.format(results['best_val_acc'])) 
    if results is not None:
        print(' Results dictionary passed  ')
    else:
        print(' Results dictionary initialization')
    if reload:
        print(' Model reloading from     : ',ckpt_file)
    else:
        print(' Model initialization ')
    print('------------------------------------------------------------------------------------------')
    
    # Measurements use for graphing loss and accuracy
    # The accuracy measured against the validation set

    
    with tf.Session() as session:
        saver = tf.train.Saver()   
         
        #  Load the weights and bias
        if reload:
            try: 
                saver.restore(session, ckpt_file)
                print(' loaded saved model checkpoint from :', ckpt_file, flush=True)
                ckpt_written = True
            except Exception as e:
                print(' Unable to load model checkpoint :',  ckpt_file)
                print(' Exception info :', e)
                raise
        else:
            ckpt_written = False
            session.run(tf.global_variables_initializer())
        


        for epoch_i in range(starting_epoch, ending_epoch + 1, 1):   
            ttl_val_accuracy = 0.0
            ttl_val_loss     = 0.0

            # Progress bar
            batches_pbar = tqdm(range(training_batches), 
                                desc='Epoch {:>2}/{} Training  '.format(epoch_i, ending_epoch), 
                                postfix={'acc': acc , 'loss': loss },
                                unit='batches', ncols =130)

            # Training cycle (batch_i)
            for _ in batches_pbar:
                # Get a batch of training features and labels
                # b_start    = batch_i * batch_size
                # b_end      = b_start + batch_size
                # trn_accuracy, trn_loss, _ = session.run([ accuracy_operation,loss_operation,training_operation], 
                                                    #   feed_dict = {x_input: x_trn[b_start : b_end], y_input: y_trn[b_start : b_end]})
 
                X_batch, y_batch = trn_datagen.next()
                
                trn_accuracy, trn_loss, _ = session.run([ accuracy_operation,loss_operation,training_operation], 
                                                      feed_dict = {x_input: X_batch, y_input: y_batch}) 

                batches_pbar.set_postfix(loss=trn_loss, acc = '{:.4f}'.format(trn_accuracy) )


                # if   (batch_i % logging_interval== 0 ):
                    # Calculate Validation accuracy
                    
                    # X_batch, y_batch = vld_datagen.next()                    
                    # val_accuracy, val_loss = session.run([accuracy_operation, loss_operation], feed_dict={x_input: X_batch, y_input: y_batch})
                    # val_accuracy, val_loss = session.run([accuracy_operation, loss_operation], feed_dict={x_input: x_vld, y_input: y_vld})
                    # epchs.append(epoch_i+1)
                    # batches.append(ttl_batches + batch_i)
                    # train_loss.append(trn_loss)
                    # train_acc.append(trn_accuracy)
                    # valid_loss.append(val_loss)
                    # valid_acc.append(val_accuracy)                    
                    
  
            ## End of Epoch operations 
            ## Calculate Validation accuracy
            validation_pbar = tqdm(range(validation_batches), 
                    desc='Epoch {:>2}/{} Validation'.format(epoch_i, ending_epoch), 
                    postfix={'acc': acc , 'loss': loss },
                    unit='batches', ncols =130)

            for _ in validation_pbar:
                X_batch, y_batch = vld_datagen.next()                    
                val_accuracy, val_loss = session.run([accuracy_operation, loss_operation], feed_dict={x_input: X_batch, y_input: y_batch})
                validation_pbar.set_postfix(loss=val_loss, acc = '{:.4f}'.format(val_accuracy) )
                ttl_val_accuracy += val_accuracy
                ttl_val_loss     += val_loss
            val_accuracy = ttl_val_accuracy / validation_batches
            val_loss     = ttl_val_loss / validation_batches

#           print("EPOCH {} ... Last Train Accuracy = {:.3f}    Last Train loss = {:.3f} ".format(epoch_i, trn_accuracy, trn_loss), flush = True)
            print("EPOCH {} ... Validation Accuracy = {:.4f}   Best Acc {:.4f}  Validation loss = {:.4f} ".format(
                    epoch_i, val_accuracy, results['best_val_acc'] , val_loss),   flush = True)
    
            ttl_batches += training_batches     ## (batch_i % log_batch_step) 

            results['epochs'   ].append(epoch_i)
            results['batches'  ].append(ttl_batches)
            results['trn_loss' ].append(trn_loss)
            results['trn_acc'  ].append(trn_accuracy)
            results['val_loss' ].append(val_loss)
            results['val_acc'  ].append(val_accuracy)
            results['learning_rate'].append(learning_rate)

            if val_accuracy > results['best_val_acc']:
                suffix = datetime.now().strftime("%y%m%d%H%M%S")
                ckpt_file = './checkpoints/'+model_config['config']+'_bs'+str(batch_size)+\
                     '_ep'+str(epoch_i)+'_dr'+str(model_config['drop_rate'])+'_'+suffix
                saver.save(session, ckpt_file)
                print(' {:7.4f}  >  {:7.4f}  - Write to checkpoint {}'.format(val_accuracy, results['best_val_acc'], ckpt_file),flush=True)
                results['last_ckpt'] = ckpt_file
                results['best_val_acc'] = val_accuracy
                results['run_ckpt_acc'].append(val_accuracy)
                results['run_epochs'].append(epoch_i)
                results['run_ckpts'].append(ckpt_file)
                ckpt_written = True

        if not ckpt_written:
            suffix = datetime.now().strftime("%y%m%d%H%M%S")
            ckpt_file = './checkpoints/'+model_config['config']+'_bs'+str(batch_size)+\
                        '_ep'+str(ending_epoch)+'_dr'+str(model_config['drop_rate'])+'_'+suffix
            saver.save(session, ckpt_file)
            results['last_ckpt'] = ckpt_file
            results['best_val_acc'] = val_accuracy            
            
            results['run_ckpt_acc'].append(val_accuracy)
            results['run_epochs'].append(ending_epoch)
            results['run_ckpts'].append(ckpt_file)
            print("Model saved to checkpoint file : ", ckpt_file, ' final epoch_i is', epoch_i)  
        else:
            print("Final checkpoint not written - better checkpoint available:  ", ckpt_file)  
    
    return results


### Define your architecture here.
### Feel free to use as many code cells as needed.

def conv2d(x, W, b , stride = 1, name = 'layer', padding = 'VALID', activation = 'RELU'):
    layer  = tf.nn.conv2d(x, W, strides = [1,stride,stride,1], padding = padding)
   
    print(' layer ', name, ' after conv2d : ', layer)
    layer  = tf.nn.bias_add(layer, b)
    print(' layer ', name, ' after bias add : ', layer)    

    if activation.upper() == 'RELU':
        layer  = tf.nn.relu(layer, name = name)
        print(' layer ', name, ' type: ', activation.upper(), ' after activation : ', layer)
    return layer


def maxpool(x, ksize = 2, stride = 2, name = 'maxpool'):
    layer_maxpool = tf.nn.max_pool(x, ksize = [1,ksize,ksize,1],strides= [1,stride,stride,1], padding = 'VALID', name = name)
    print(' layer ', name, ' after max pooling : ', layer_maxpool)    
    return layer_maxpool


def fully_connected(x, W, b, name = 'fc', activation = None):
    layer = tf.add(tf.matmul(x, W), b, )
    print(' layer ', name, ' before activation : ', layer)
    
    if activation is not None :

        if activation.upper() == 'RELU':
            layer  = tf.nn.relu(layer, name = name)
            print(' layer ', name, ' type: ', activation.upper(), ' after activation : ', layer)
    
    return layer
    
def dropout(x, drop_rate = 0.00, name = 'dropout'):
    layer = tf.nn.dropout(x, rate = drop_rate, name = name)
    return layer 

def model(x, model_config, mu = 0, sigma = 0.1, drop_rate = 0.00, debug = False):

    weights   = model_config['weights']
    biases    = model_config['biases']    
    drop_rate = model_config['drop_rate']
    
    if debug:
        print(weights, biases)
    
    layer1 = conv2d(x, weights['W1'], biases['W1'], name = 'layer1')
    maxpool1 = maxpool(layer1, ksize=2, stride =2, name='maxpool1')
    
    layer2 = conv2d(maxpool1, weights['W2'], biases['W2'], name = 'layer2')
    maxpool2 = maxpool(layer2, ksize=2, stride =2, name='maxpool2')
    
    flattened = tf.keras.layers.Flatten()(maxpool2)
    print(' flattened after flattened() : ', flattened)  
    
    fc1 = fully_connected(flattened, weights['FC1'], biases['FC1'], name = 'FC1', activation= 'relu')
    fc1_dropout = dropout(fc1, drop_rate = drop_rate, name ='fc1dropout')
 
    fc2 = fully_connected(fc1_dropout, weights['FC2'], biases['FC2'], name = 'FC2', activation= 'relu')
    fc2_dropout = dropout(fc2, drop_rate = drop_rate, name = 'fc2dropout')    
    
    logits = fully_connected(fc2_dropout, weights['FC3'], biases['FC3'], name = 'logits')
    
    return logits


def model_accuracy(logits, y_onehot):
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_onehot, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return accuracy_operation
    
    
def model_optimization(logits, y_onehot , LR = 0.1):

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot, logits=logits)
    loss_op = tf.reduce_mean(cross_entropy, name = 'loss')

    training_op = tf.train.AdamOptimizer(learning_rate = LR).minimize(loss_op)    
    
    return loss_op, training_op


    

def def_weights_biases(model_config):

    mu        = model_config.get('mu', 0) 
    sigma     = model_config.get('sigma', 0.1)
    f1_size   = model_config.get('f1_size'   , 5)
    f2_size   = model_config.get('f2_size'   , 5)
    l1_units  = model_config.get('l1_units'  , 16)
    l2_units  = model_config.get('l2_units'  , 16)
    fc1_units = model_config.get('fc1_units' , 240)
    fc2_units = model_config.get('fc2_units' , 240)
    fc3_units = model_config.get('fc3_units' , 43)

    model_config['weights'] = {
        'W1' : tf.Variable(tf.truncated_normal([f1_size,f1_size,3,l1_units], mean = mu, stddev = sigma), name = 'CONV1_W', dtype=tf.float32),
        'W2' : tf.Variable(tf.truncated_normal([f2_size,f2_size,l1_units,l2_units], mean = mu, stddev = sigma), name = 'CONV2_W', dtype=tf.float32),
        'FC1' : tf.Variable(tf.truncated_normal([f2_size*f2_size*l2_units, fc1_units], mean = mu, stddev = sigma), name = 'FC1_W', dtype=tf.float32),
        'FC2' : tf.Variable(tf.truncated_normal([fc1_units,fc2_units], mean = mu, stddev = sigma), name = 'FC2_W', dtype=tf.float32),
        'FC3' : tf.Variable(tf.truncated_normal([fc2_units,fc3_units], mean = mu, stddev = sigma), name = 'FC3_W', dtype=tf.float32)
    }

    model_config['biases'] = {
        'W1' : tf.Variable(tf.truncated_normal([l1_units], mean = mu, stddev = sigma), name = 'CONV1_B', dtype=tf.float32),
        'W2' : tf.Variable(tf.truncated_normal([l2_units], mean = mu, stddev = sigma), name = 'CONV2_B', dtype=tf.float32),
        'FC1' : tf.Variable(tf.truncated_normal([fc1_units], mean = mu, stddev = sigma), name = 'FC1', dtype=tf.float32),
        'FC2' : tf.Variable(tf.truncated_normal([fc2_units], mean = mu, stddev = sigma), name = 'FC2', dtype=tf.float32),
        'FC3' : tf.Variable(tf.truncated_normal([fc3_units], mean = mu, stddev = sigma), name = 'FC3', dtype=tf.float32)
    }    


'''
def evaluate(sess, x_data, y_data, batch_size, eval_ops):
    accuracy_op, loss_op = eval_ops
    
    num_examples = len(x_data)
    
    total_accuracy = 0
    total_loss     = 0 
    sess = tf.get_default_session()
    
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = x_data[offset:offset+ batch_size], y_data[offset:offset+batch_size]
        
        accuracy_np, loss_np = sess.run([accuracy_op, loss_op], feed_dict={x_input: batch_x, y_input: batch_y})
        
        total_accuracy += (accuracy_np * len(batch_x))
        total_loss     += (loss_np     * len(batch_x))
    
    total_accuracy /= num_examples
    total_loss     /= num_examples

    return  total_loss,total_accuracy   
'''
    return