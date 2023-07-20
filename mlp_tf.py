import argparse
import json
import pickle
import numpy as np
from json_loader import json_to_dataset
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras import losses
from keras import optimizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime as dt
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from timer import Timer
import matplotlib.pyplot as plt


# FOR DEV: pkl cache ==============================================================
def pickle_cache(data, cache_path, pkl_filename):
    file_path = os.path.join(cache_path, pkl_filename)
    print("saving matched traj to pkl ...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("Successfully saved matched traj to %s."%file_path)


def pickle_load(path):
    print("loading matched traj from pkl ...")
    with open(path, 'rb') as f:
        return pickle.load(f)


# Class Multi-Layer-Perceptron =========================================================
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

        loss_fn = losses.BinaryCrossentropy()
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

# Utils ================================================================================
def cvt_nparray(dataset, k_item):
    print("Converting feature vectors into nparray ...")
    timer = Timer()
    timer.start()
    arr = np.array([dataset[0][k_item]])
    for data in dataset[1:]:
        arr = np.concatenate((arr, [data[k_item]]), axis=0)
    timer.stop()
    return arr

# def evaluate(pred_ls, gt_ls, th_for_true):
#     pred_b_ls = [y_pred > th_for_true for y_pred in pred_ls]
#     pred_t_count = sum(pred_b_ls)
#     gt_b_ls = [y_gt > th_for_true for y_gt in gt_ls]
#     gt_t_count = sum(gt_b_ls)
#     print("[Eval] Sum of T\n\t- in PRED:\t %d;\n\t- in GT:\t %d"%(pred_t_count, gt_t_count))

#     tp = 0
#     fp = 0
#     fn = 0
#     total = len(gt_b_ls)
#     for i in range(total):
#         if pred_b_ls[i] and gt_b_ls[i]:
#             tp += 1
#         if pred_b_ls[i] and not gt_b_ls[i]:
#             fp += 1
#         if not pred_b_ls[i] and gt_b_ls[i]:
#             fn += 1
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1 = 2*(precision*recall)/(precision+recall)
#     print("[Eval] Metrics:\n\t- Precision:\t %.5f\n\t- Recall:\t %.5f\n\t- F1 score:\t %.5f"%(precision, recall, f1))
#     return precision, recall, f1

def evaluate_sklearn(pred_ls, gt_ls, th_for_true):
    pred_b_ls = [y_pred > th_for_true for y_pred in pred_ls]
    pred_t_count = sum(pred_b_ls)
    gt_b_ls = [y_gt > th_for_true for y_gt in gt_ls]
    gt_t_count = sum(gt_b_ls)
    print("[Eval] Sum of T\n\t- in PRED:\t %d;\n\t- in GT:\t %d"%(pred_t_count, gt_t_count))

    eval_metrics = {
        "accuracy": accuracy_score(gt_b_ls, pred_b_ls),
        "precision": precision_score(gt_b_ls, pred_b_ls),
        "recall": recall_score(gt_b_ls, pred_b_ls),
        "f1": f1_score(gt_b_ls, pred_b_ls)
    }

    print("[Eval] Sklearn Metrics:\n\t- Accuracy:\t %.5f\n\t- Precision:\t %.5f\n\t- Recall:\t %.5f\n\t- F1 score:\t %.5f"\
              %(eval_metrics['accuracy'], eval_metrics['precision'], eval_metrics['recall'],eval_metrics['f1']))
    return eval_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--is_dataset_pkl', default=True, # whether load dataset from pkl
                        help='load pre-processed data from pkl')
    parser.add_argument('--json_folder', default='./labels/', 
                        help='path of the json file containing labels')
    parser.add_argument('--mlp_mode', default="pred", # MLP mode
                        help='MUST BE "train" OR "pred"')
    parser.add_argument('--model_path', default="saved_models/tf/20230719_174523_ep128_bs64.h5", 
                        help='path to an existing .h5 model file')
    args = parser.parse_args()
    print('Start with the args: {}'.format(args))

    # Load configs
    configs = json.load(open('configs.json', 'r'))

    json_folder = "labels"
    fname = "20210609_123753_BB_split_000"
    json_fname = "%s.json"%fname

    if not args.is_dataset_pkl:
        # Load data ################################################################################ 
        cam_train_set, cam_test_set = json_to_dataset(json_folder, json_fname, configs, sensor="camera")

        x_train = cam_train_set['x']
        y_train = cam_train_set['y']
        x_test = cam_test_set['x']
        y_test = cam_test_set['y']

        # Save pkl
        dataset_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test 
        }
        pickle_cache(dataset_dict, "cache_dataset", "dataset_%s.pkl"%fname)
    else:
        # Load pkl ################################################################################
        loaded_dict = pickle_load("cache_dataset/dataset_%s.pkl"%fname)
        x_train = loaded_dict['x_train']
        y_train = loaded_dict['y_train']
        x_test = loaded_dict['x_test']
        y_test = loaded_dict['y_test']

    # Model
    mlp = MLP()
    if args.mlp_mode == "train":
        print("\n[Model] Mode: TRAIN\n")
        # Train
        mlp.build_model(configs)
        mlp.train(x_train, y_train)
        # Eval
        mlp.eval(x_test, y_test)
        mlp.disp_history()

    if args.mlp_mode == "pred":
        print("\n[Model] Mode: PRED\n")
        # Pred
        model_path = args.model_path
        mlp.load_model(model_path)
    
        # >>>>> deploy to display later
        y_pred = mlp.pred_single_sample(x_test[0])
        y_pred_b = mlp.pred_single_sample(x_test[0], b_cvt_tf=True, th=configs['conf_th'])

        # >>>>> evaluate sklearn:
        y_pred_batch = mlp.pred_batch_samples(x_test)
        eval_metrics = evaluate_sklearn(y_pred_batch, y_test, configs['conf_th'])

    print("End")
