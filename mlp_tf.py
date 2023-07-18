import argparse
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras import losses
from keras import optimizers
import datetime as dt
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sample_loader import *
from timer import Timer
import matplotlib.pyplot as plt


class MLP():
    def __init__(self):
        self.model = Sequential()
        self.history = None
        self.configs = {}
        self.eval_result = {}
    
    def load_model(self, model_path):
        print("[Model] Loading model from file %s"%model_path)
        self.model = load_model(model_path)

    def pred_single_sample(self, x_single, b_cvt_tf=False, th=None):
        out = self.model.predict(np.array([x_single]))[0, 0]
        if b_cvt_tf and th is not None:
            out = True if out > th else False
        return out
    
    def pred_batch_samples(self, x_batch):
        return np.squeeze(self.model.predict(np.array(x_batch)))

    def build_model(self, model_configs):
        self.configs = model_configs
        self.model.add(Dense(self.configs['hidden_dims'][0], 
                             input_dim=self.configs['input_size'],
                             kernel_initializer=tf.initializers.he_uniform()))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        for h_dim in self.configs['hidden_dims'][1:]:
            self.model.add(Dense(h_dim, kernel_initializer=tf.initializers.he_uniform()))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            # self.model.add(Dropout(0.2))
        self.model.add(Dense(self.configs['num_classes'], 
                             kernel_initializer=tf.initializers.glorot_uniform()))
        self.model.add(Activation('sigmoid'))

        # loss_fn = losses.SparseCategoricalCrossentropy()
        loss_fn = losses.BinaryCrossentropy()
        # optimizer = optimizers.Adam(learning_rate=self.configs['lr'])
        optimizer = optimizers.RMSprop(learning_rate=self.configs['lr'])

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        print('[Model] Model Compiled')
        self.model.summary()

    def train(self, x, y, b_save_model=True):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.configs['num_epochs'], self.configs['batch_size']))
        save_fname = os.path.join(self.configs['save_dir_tf'], '%s_ep%s_bs%s.h5'\
                                  % (dt.datetime.now().strftime('%Y%m%d_%H%M%S'),\
                                     str(self.configs['num_epochs']),\
                                     str(self.configs['batch_size'])))
        callbacks = [
			# EarlyStopping(monitor='val_loss', patience=5),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=False)
		]
        self.history = self.model.fit(
            x,
            y,
            batch_size=self.configs['batch_size'],
            epochs=self.configs['num_epochs'],
            callbacks=callbacks,
            verbose=1,
            validation_split=0.2
        )
        if b_save_model:
            self.model.save(save_fname)
            print('[Model] Training Completed. Model saved as %s' % save_fname)

    def disp_history(self):
        loss_values = self.history.history['loss']
        val_loss_values = self.history.history['val_loss']
        acc_values = self.history.history['accuracy']
        val_acc_values = self.history.history['val_accuracy']

        fig, ax1 = plt.subplots()
        fig.suptitle('%s\nepochs %d\nbatch_size %d\ntest_acc %.4f'\
                      %(dt.datetime.now().strftime('%Y%m%d_%H:%M:%S'),
                      self.configs['num_epochs'], 
                      self.configs['batch_size'],
                      self.eval_result['test_acc']))

        ax1.plot(loss_values, label='Train Loss', color='tab:red')
        ax1.plot(val_loss_values, label='Val Loss', color='tab:orange')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.set_ylim(0, 2)
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        ax2.plot(acc_values, label='Train Accuracy', color='tab:blue')
        ax2.plot(val_acc_values, label='Validation Accuracy', color='tab:green')
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        ax1.text(0.5, 0.5, "Test loss: %.4f\nTest acc : %.2f"%\
                 (self.eval_result['test_loss'], 100*self.eval_result['test_acc']),\
                 ha='left', va='bottom', transform=ax1.transAxes, fontsize=12)
        plt.show()

    def eval(self, x_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(x=x_test, y=y_test, verbose=0)
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        self.eval_result = {
            'test_loss': test_loss,
            'test_acc': test_accuracy
        }

def cvt_nparray(dataset, k_item):
    print("Converting feature vectors into nparray ...")
    timer = Timer()
    timer.start()
    arr = np.array([dataset[0][k_item]])
    for data in dataset[1:]:
        arr = np.concatenate((arr, [data[k_item]]), axis=0)
    timer.stop()
    return arr

def evaluate(pred_b_arr, gt_b_arr):
    tp = 0
    fp = 0
    fn = 0
    total = len(gt_b_arr)
    for i in range(total):
        if pred_b_arr[i] and gt_b_arr[i]:
            tp += 1
        if pred_b_arr[i] and not gt_b_arr[i]:
            fp += 1
        if not pred_b_arr[i] and gt_b_arr[i]:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--is_dataset_pkl', default=True, 
                        help='load pre-processed data from pkl')
    parser.add_argument('--json_folder', default='./labels/', 
                        help='path of the json file containing labels')
    parser.add_argument('--is_train', default=False, 
                        help='train an MLP model from samples')
    parser.add_argument('--is_pred', default=True, 
                        help='load an existing MLP model to do prediction')
    parser.add_argument('--model_path', default="saved_models/tf/20230707_143235_ep128_bs64.h5", 
                        help='path to an existing .h5 model file')
    args = parser.parse_args()
    print('Start with the args: {}'.format(args))

    # Load configs
    configs = json.load(open('configs.json', 'r'))
    model_configs = configs['model_configs']
    data_configs = configs['data_configs']

    # dataset_name = "BB123753"
    dataset_name = "BB"
    # dataset_name = "full"

    if not args.is_dataset_pkl:
        # Load data ################################################################################ 
        folder = "labels"
        files_ls = [
            "label_20210609_122855_BB_split_000.json",
            "label_20210609_123753_BB_split_000.json"
        ]
        
        cam_dataset, lrr_dataset = read_jsons(folder, files_ls)
        cam_train_set, cam_test_set = samples_to_train(cam_dataset, 
                                                       seq_len=data_configs['seq_len'], 
                                                       split_ratio=data_configs['split_ratio'], 
                                                       batch_size=model_configs['batch_size'], 
                                                       input_size=model_configs['input_size'])

        x_train = cam_train_set['x']
        y_train = cam_train_set['y']
        x_test = cam_test_set['x']
        y_test = cam_test_set['y']

        # Save pkl
        pickle_cache(x_train, "cache_dataset", "%s_x_train.pkl"%dataset_name)
        pickle_cache(y_train, "cache_dataset", "%s_y_train.pkl"%dataset_name)
        pickle_cache(x_test, "cache_dataset", "%s_x_test.pkl"%dataset_name)
        pickle_cache(y_test, "cache_dataset", "%s_y_test.pkl"%dataset_name)
    else:
        # Load pkl ################################################################################
        x_train = pickle_load("cache_dataset/%s_x_train.pkl"%dataset_name)
        y_train = pickle_load("cache_dataset/%s_y_train.pkl"%dataset_name)
        x_test = pickle_load("cache_dataset/%s_x_test.pkl"%dataset_name)
        y_test = pickle_load("cache_dataset/%s_y_test.pkl"%dataset_name)

    # Model
    mlp = MLP()
    if args.is_train:
        # Train
        mlp.build_model(model_configs)
        mlp.train(x_train, y_train)
        # Eval
        mlp.eval(x_test, y_test)
        mlp.disp_history()
    if args.is_pred:
        # Pred
        model_path = args.model_path
        mlp.load_model(model_path)
    
        # >>>>> deploy to display later
        y_pred = mlp.pred_single_sample(x_test[0])
        y_pred_b = mlp.pred_single_sample(x_test[0], b_cvt_tf=True, th=data_configs['conf_th'])

        # >>>>> TEST: convert result 0/1 back to F/T
        y_pred_batch =mlp.pred_batch_samples(x_test)
        conf_th = 0.80

        y_pred_batch_bool = [y_pred > conf_th for y_pred in y_pred_batch]
        pred_t_count = sum(y_pred_batch_bool)

        y_gt_batch_bool = [y_gt > conf_th for y_gt in y_test]
        gt_t_count = sum(y_gt_batch_bool)
        print("[Eval] Num of T in PRED: %d; in GT: %d"%(pred_t_count, gt_t_count))

        # >>>>> evaluate
        precision, recall = evaluate(y_pred_batch_bool, y_gt_batch_bool)
        print("[Eval] Metrics:\nPrecision: %.3f\nRecall: %.3f"%(precision, recall))

    print("End")
