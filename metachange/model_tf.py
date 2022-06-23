import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
import gc
from .model_np import Model_dist
from .model_np import Model_acc

def show_batch(batch_img, batch_label):
    N = batch_img.shape[0]
    n_col = 8
    n_row = (N+n_col-1)//n_col
    
    #print(N, n_row, n_col)
    #print(batch_img.shape)
    
    fig=plt.figure(figsize=(8*1.6, n_row*2.0))
    for i in range(N):
        axi = fig.add_subplot(n_row, n_col, i+1)
        img_i = batch_img[i]
        
        cmap = "gray"
        if(img_i.shape[2]==1): 
            img_i = img_i[:,:,0]
            cmap = "gray"

        plt.imshow(img_i, cmap=cmap, vmin=0, vmax=1.0)
        
        axi.axis('off')
        axi.set_title('y = %d'%(batch_label[i]))
    plt.show()

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=0)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    # return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    # return tf.reshape(img, (IMG_HEIGHT, IMG_WIDTH))
    return img

def process_path(file_path):
    #label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def dataset_from_file(mode, file_pattern, t_cut):
    if(mode == "pattern"):
        fname_list = glob.glob(file_pattern)
        picid = [s.replace("/",".").split(".")[-2] for s in fname_list]
        picid.sort()
        list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    elif(mode == "list"):
        fname_list = file_pattern
        picid = [s.replace("/",".").split(".")[-2] for s in fname_list]
        list_ds = tf.data.Dataset.from_tensor_slices(fname_list)
    else:
        raise Exception("Wrong model, only pattern or list allowed.")
    
    img_ds = list_ds.map(process_path, num_parallel_calls=None)
    
    #pos_slash = file_pattern.rfind('/')
    
    pos_slash = fname_list[0].rfind('/')
    fname_meta = fname_list[0][:pos_slash]+"/meta_data.bin"
    meta = pickle.load(open(fname_meta, "rb"))
    labels = [1 if meta[i]["t"]>t_cut else 0 for i in picid]
    label_ds = tf.data.Dataset.from_tensor_slices(tf.dtypes.cast(labels, tf.int64))
    
    train_data = tf.data.Dataset.zip((img_ds, label_ds))
    
    #train_data = train_data.cache()
    
    return train_data

def get_image_and_indicator(file_pattern):
    fname_list = glob.glob(file_pattern)
    picid = [s.replace("/",".").split(".")[-2] for s in fname_list]
    picid.sort()
    
    #print(picid[:10])
    list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    img_ds = list_ds.map(process_path, num_parallel_calls=None)
    
    pos_slash = file_pattern.rfind('/')
    fname_meta = file_pattern[:pos_slash]+"/meta_data.bin"
    meta = pickle.load(open(fname_meta, "rb"))
    
    #labels = [1 if meta[i]["t"]>t_cut else 0 for i in picid]
    #label_ds = tf.data.Dataset.from_tensor_slices(tf.dtypes.cast(labels, tf.int64))
    
    t = [meta[i]["t"] for i in picid]
    t_ds = tf.data.Dataset.from_tensor_slices(tf.dtypes.cast(t, tf.float32))
    
    #train_data = tf.data.Dataset.zip((img_ds, label_ds))
    
    train_data = train_data.cache()
    
    return tf.data.Dataset.zip((img_ds, t_ds))

def get_image_label(ds_zip, t_cut):
    print(ds_zip)
    print(t_cut)
    
    def t_to_label(img, t):
        b = tf.cond(tf.math.greater(t, tf.constant(t_cut, dtype=tf.float32)),\
                        lambda:tf.constant(1, dtype=tf.int64),\
                        lambda:tf.constant(0, dtype=tf.int64))
        return (img, b)
    
    train_data = ds_zip.map(t_to_label, num_parallel_calls=None)
    
    #train_data = tf.data.Dataset.zip((image_ds, label_ds))
    return train_data

def dataset_from_file_exp(file_pattern, t_cut):
    ds_zip = get_image_and_indicator(file_pattern)
    return get_image_label(ds_zip, t_cut)

def conv2d(input, kernel_size, stride, num_filter):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

    W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
    b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
    return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

def max_pool(input, kernel_size, stride):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

def flatten(input):
    """
        - input: input tensors
    """
    return tf.reshape(input, [-1,np.prod(input.get_shape()[1:])])

def fc(input, num_output):
    """
        - input: input tensors
        - num_output: int, the output dimension
    """
    return tf.layers.dense(input, num_output, activation=None)

def norm(input, is_training):
    """
        - input: input tensors
        - is_training: boolean, if during training or not
    """
    return tf.layers.batch_normalization(input, training=is_training)


class Model(object):
    def __init__(self, imsize, batch_size = 128, n_hidden=64):
        self.batch_size = batch_size
        self.imsize = imsize
        self.n_hidden = n_hidden
        self.n_class = 2
        self.log_step = 10
        self.shuffle_size = 40000
        
        self.verbos = True
        self.ifdo_valid = False
        self.ifdo_test = False
        
        self.data = {}
        self.data["train"] = None
        self.data["valid"] = None
        self.data["test"] = None
        
        self._build_model()
    
    def set_data_target(self, name, dataset):
        if(name not in self.data): raise NameError('Illegal dataset name %s'(name))
        self.data[name] = dataset.shuffle(self.shuffle_size).batch(self.batch_size)
    
    def _build_optimizer(self):
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-4
        
        
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           500, 0.96, staircase=False)
        
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_op, global_step=global_step)
        
    
    def _build_model(self):
        
        ## define the dimensions of x and y
        #placeholder_x = tf.placeholder(tf.float32, [None, self.imsize[0], self.imsize[1], self.imsize[2]])
        #placeholder_y = tf.placeholder(tf.int64, [None])
        #sample_dataset = tf.data.Dataset.from_tensor_slices((placeholder_x,placeholder_y)).batch(self.batch_size)
        #print(sample_dataset.output_types)
        #print(sample_dataset.output_shapes)
        
        ## define the input iterator as reinitilizable
        self.input_iter = tf.data.Iterator.from_structure(\
            (tf.float32, tf.int64),\
            (tf.TensorShape([None, self.imsize[0], self.imsize[1], self.imsize[2]]), tf.TensorShape([None]))\
            )
        
        ## the input data
        self.X, self.Y = self.input_iter.get_next()
        
        labels = tf.one_hot(self.Y, self.n_class)
        
        ## the logit scores output by nn
        self.logits = self._model()
        
        ## the loss op
        self.loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = self.logits))
        
        ## boolean var to mark whether being trained
        self.istraining = tf.placeholder(tf.bool)
        
        ## predictions and accuracy
        self.predicts = tf.argmax(self.logits, 1)
        correct = tf.equal(self.predicts, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        ## optimizer
        self._build_optimizer()
    
    def _model(self):
        if(self.verbos): print('intput layer: ' + str(self.X.get_shape()))
        
        with tf.variable_scope('conv1'):
            self.conv1 = conv2d(self.X, 3, 1, 32)
            self.relu1 = tf.nn.relu(self.conv1)
            #self.norm1 = norm(self.relu1, self.istraining)
            if(self.verbos): print('conv1 layer: ' + str(self.relu1.get_shape()))
            
        with tf.variable_scope('conv2'):
            self.conv2 = conv2d(self.relu1, 3, 1, 32)
            self.relu2 = tf.nn.relu(self.conv2)
            #self.norm2 = norm(self.relu2, self.istraining)
            self.pool2 = max_pool(self.relu2, 2, 2)
            self.drop2 = tf.nn.dropout(self.pool2, 0.2)
            if(self.verbos): print('conv2 layer: ' + str(self.drop2.get_shape()))
        
        with tf.variable_scope('conv3'):
            self.conv3 = conv2d(self.drop2, 3, 1, 64)
            self.relu3 = tf.nn.relu(self.conv3)
            #self.norm3 = norm(self.relu3, self.istraining)
            if(self.verbos): print('conv3 layer: ' + str(self.relu3.get_shape()))
        
        with tf.variable_scope('conv4'):
            self.conv4 = conv2d(self.relu3, 3, 1, 64)
            self.relu4 = tf.nn.relu(self.conv4)
            #self.norm4 = norm(self.relu4, self.istraining)
            self.pool4 = max_pool(self.relu4, 2, 2)
            self.drop4 = tf.nn.dropout(self.pool4, 0.2)
            if(self.verbos): print('conv4 layer: ' + str(self.drop4.get_shape()))
        
        with tf.variable_scope('conv5'):
            self.conv5 = conv2d(self.drop4, 3, 1, 128)
            self.relu5 = tf.nn.relu(self.conv5)
            #self.norm5 = norm(self.relu5, self.istraining)
            if(self.verbos): print('conv5 layer: ' + str(self.relu5.get_shape()))
        
        with tf.variable_scope('conv6'):
            self.conv6 = conv2d(self.relu5, 3, 1, 128)
            self.relu6 = tf.nn.relu(self.conv6)
            #self.norm6 = norm(self.relu6, self.istraining)
            self.pool6 = max_pool(self.relu6, 2, 2)
            self.drop6 = tf.nn.dropout(self.pool6, 0.2)
            if(self.verbos): print('conv6 layer: ' + str(self.drop6.get_shape()))
            
        self.flat = flatten(self.drop6)     
        if(self.verbos): print('flat layer: ' + str(self.flat.get_shape()))
        
        with tf.variable_scope('fc'):
            self.fc = fc(self.flat, self.n_class)
            if(self.verbos): print('fc layer: ' + str(self.fc.get_shape()))
        
        return self.fc
    
    def train(self, sess, n_epoch):
        sess.run(tf.global_variables_initializer())
        
        if(self.data["train"]==None):
            raise ValueError('Training dataset not set!')
        
        step = 0
        losses = []
        accuracies = []
        
        ## for every epoch, shuffle the training dataset
        #self.data["train"] = self.data["train"].shuffle(self.shuffle_size)
        
        if(self.verbos): print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(n_epoch):
            if(self.verbos): print('train for epoch %d' % epoch)
            
            ## initialize the input interator as data[train]
            sess.run(self.input_iter.make_initializer(self.data["train"]))
            
            while(True):
                try:
                    feed_dict = {self.istraining:True}
                    fetches = [self.train_op, self.loss_op, self.accuracy_op, self.Y]
                    _, loss, accuracy, Y = sess.run(fetches, feed_dict=feed_dict)
                    loss = loss/Y.shape[0]
                    losses.append(loss)
                    accuracies.append(accuracy)
                    if step % self.log_step == 0:
                        if(self.verbos): print('iteration (%8d): loss = %8.4f, accuracy = %5.4f' %
                            (step, loss, accuracy))
                    step += 1
                except tf.errors.OutOfRangeError:
                    ## one epoch is over, no more batches
                    break
            
            if(self.ifdo_valid): self.evaluate(sess, "valid")
            if(self.ifdo_test): self.evaluate(sess, "test")
                    
        if(self.verbos): print("Training complete.")
        
        if(self.verbos):
            #plt.plot(np.arange(len(losses))*1.*n_epoch/len(losses), losses)
            plt.semilogy(np.arange(len(losses))*1.*n_epoch/len(losses), losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()
        
        if(self.verbos):
            sess.run(self.input_iter.make_initializer(self.data["train"]))
            feed_dict = {self.istraining:False}
            sample_X, sample_Y = sess.run([self.X, self.Y])
            show_batch(sample_X, sample_Y)
    
    def evaluate(self, sess, name):
        if(self.verbos): print('-' * 5 + '  Evaluating  ' + '-' * 5)
        if(name not in self.data): raise NameError('Illegal dataset name %s'(name))
        if(self.verbos): print("Dataset used: %s"%(name))
        if(self.data[name]==None): raise ValueError('Evaluation dataset not set!')
        
        ## switch to the defined dataset
        sess.run(self.input_iter.make_initializer(self.data[name]))
        
        losses = []
        accuracies = []
        
        #maj_list = []
        cont_1 = 0
        cont_tot = 0
        
        step = 0
        while(True):
            try:
                feed_dict = {self.istraining:False}
                fetches = [self.loss_op, self.accuracy_op, self.Y]
                loss, accuracy, Y = sess.run(fetches, feed_dict=feed_dict)
                loss = loss/Y.shape[0]
                losses.append(loss)
                accuracies.append(accuracy)
                
                #maj_list.append(np.maximum(1.-np.mean(Y), np.mean(Y)))
                cont_1 += np.sum(Y)
                cont_tot += Y.shape[0]
                
                if step % self.log_step == 0:
                    if(self.verbos): print('iteration (%8d): loss = %8.4f, accuracy = %5.4f'%
                        (step, loss, accuracy))
                step += 1
            except tf.errors.OutOfRangeError:
                ## one epoch is over, no more batches
                break
        mean_acc = np.mean(np.array(accuracies))
        mean_loss = np.mean(np.array(losses))
        
        #mean_maj = np.mean(np.array(maj_list))
        mean_maj = np.maximum(cont_1*1.0/cont_tot, 1-cont_1*1.0/cont_tot)
        
        if(self.verbos): print("mean accuracy = %.4f"%(mean_acc))
        if(self.verbos): print("maj class = %.4f"%(mean_maj))
        return {"acc":mean_acc, "maj":mean_maj, "accdev":mean_acc-mean_maj}


def train_random_split_tf(folder_name, num_point=20, n_trial=6, n_epoch=10, n_batch=16, clear_res=False):
    print(folder_name)
    if(clear_res): os.system("rm -rf ./%s/res_folder; mkdir ./%s/res_folder"%(folder_name, folder_name))
    if(not os.path.exists("./%s/res_folder"%(folder_name))):
        print("mkdir ./%s/res_folder"%(folder_name))
        os.system("mkdir ./%s/res_folder"%(folder_name))
    
    train_valid_files = folder_name + "/dataset_train/*.png"
    test_files = folder_name + "/dataset_test/*.png"

    train_valid_file_list = glob.glob(train_valid_files)
    test_file_list = glob.glob(test_files)

    n_train_valid = len(train_valid_file_list)
    n_test = len(test_file_list)
    n_total = n_train_valid+n_test 
    n_train = int(n_total*0.5)
    n_valid = n_train_valid - n_train
    print("n_total = ", n_total)
    print("n_train = %d, n_valid = %d, n_test = %d"%(n_train, n_valid, n_test))

    meta_file = folder_name+"/dataset_train/meta_data.bin"
    meta_data = pickle.load(open(meta_file, "rb"))
    t_values = np.array([meta_data[i]['t'] for i in meta_data])
    t_min, t_max = np.min(t_values), np.max(t_values)
    t_list = np.linspace(t_min, t_max, num=num_point, endpoint=True)


    for i in range(n_trial):
        print("Trial #%d:"%(i))
        acc_list = []
        maj_list = []
        accdev_list = []
        
        fname_i = "./%s/res_folder/res_%04d.bin"%(folder_name,i)
        if(os.path.exists(fname_i)):
            print("Skipped!")
            continue
        
        np.random.shuffle(train_valid_file_list)
        train_file_list = train_valid_file_list[:n_train]
        valid_file_list = train_valid_file_list[n_train:]

        res_curve = {}

        for ti in t_list:
            ## a training data point
            tf.reset_default_graph()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                data_train = dataset_from_file("list", train_file_list, ti)
                data_valid = dataset_from_file("list", valid_file_list, ti)
                data_test = dataset_from_file("list", test_file_list, ti)
                
                #data_train = get_image_label(ds_train, ti)
                #data_valid = get_image_label(ds_valid, ti)
                #data_test = get_image_label(ds_test, ti)
                
                model = Model((64,64,1), batch_size = n_batch, n_hidden=64)

                model.set_data_target("train", data_train)
                model.set_data_target("valid", data_valid)
                model.set_data_target("test", data_test)
                #model.verbos=True
                model.train(sess, n_epoch)
                res_train = model.evaluate(sess, "test")

                if(True):
                    print("ti = %.4e, acc = %.4f, maj = %.4f, accdev = %.4f"%\
                          (ti, res_train["acc"], res_train["maj"], res_train["accdev"]))

                acc_list.append(res_train["acc"])
                maj_list.append(res_train["maj"])
                accdev_list.append(res_train["accdev"])
        res_curve["lbd"] = t_list
        res_curve["acc"] = acc_list
        res_curve["maj"] = maj_list
        res_curve["accdev"] = accdev_list
        fname_i = "./%s/res_folder/res_%04d.bin"%(folder_name,i)    
        pickle.dump(res_curve, open(fname_i, 'wb'))
        gc.collect()


def meta_change_from_file(path):
    ## load t
    fname_meta = path + "/dataset_train/meta_data.bin"
    meta = pickle.load(open(fname_meta, "rb"))
    t_train = [meta[pic_id]['t'] for pic_id in meta]
    
    fname_meta = path + "/dataset_test/meta_data.bin"
    meta = pickle.load(open(fname_meta, "rb"))
    t_test = [meta[pic_id]['t'] for pic_id in meta]
    
    t = t_train + t_test
    print("number t = ", len(t))
    
    t = np.array(t)
    #t = np.random.uniform(0.0, 1.0, 4000)
    model_dist = Model_dist(t, 100)
    model_acc = Model_acc(model_dist)
    
    alpha_i = []
    t0_i = []

    res_files = sorted(glob.glob(path+"/res_folder/*.bin"))
    n_trial = len(res_files)

    res = []
    for rf_i in res_files:

        res_trial = pickle.load(open(rf_i, "rb"))

        res_trial["acc"] = np.maximum(res_trial["acc"], res_trial["maj"])
        
        fit_res = model_acc.fit(res_trial)
        print("t0 = %.4f, alpha = %.4f"%(fit_res["t0"], fit_res["alpha"]))

        ## add infer results
        n_seg_model = 100
        t_left = np.min(t_train)
        t_right = np.max(t_train)
        model_t_list = np.linspace(t_left, t_right, n_seg_model, endpoint=True)
        model_curve = model_acc.query_acc_list(model_t_list)["accdev"]

        res_trial["model_t0"] = fit_res["t0"]
        res_trial["model_alpha"] = fit_res["alpha"]
        res_trial["model_curve"] = [model_t_list, model_curve]

        res.append(res_trial)

    model_t0_all = np.array([res[i]["model_t0"] for i in range(n_trial)])
    model_alpha_all = np.array([res[i]["model_alpha"] for i in range(n_trial)])
    
    print("t0 = %.4f +- %.4f"%(np.mean(model_t0_all), np.std(model_t0_all)))
    print("alpha = %.4f +- %.4f"%(np.mean(model_alpha_all), np.std(model_alpha_all)))
    
    return res