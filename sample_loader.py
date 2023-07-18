import json
import os
import random
import numpy as np
from timer import Timer
import pickle


def get_files_in_folder(folder_path):
    files_ls = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            files_ls.append(file_name)
    return files_ls

def read_jsons(folder_path, files_ls=None):
    """
    Read batch of .json files from folder,
    split camera and radar data
    """
    cam_ls = []
    lrr_ls = []

    files_ls = files_ls if files_ls is not None else get_files_in_folder(folder_path)

    print("[Dataset] %d json files found."%len(files_ls))
    for fname in files_ls:
        timer = Timer()
        print("\n[Dataset] processing %s"%fname)
        timer.start()
        data_ls = json.load(open(os.path.join(folder_path, fname), 'r'))
        timer.stop()

        len_json = len(data_ls)
        len_data = 0
        for i, data in enumerate(data_ls):
            if data['actor_traj'][0]['global'] == data['ego_traj'][0]['global']:
                if data['actor_traj'][0]['sensor'] == "camera":
                    cam_ls.append({
                        'global': data['ego_traj'][0]['global'],
                        'actor_id':data['actor_traj'][0]['id'],
                        'ro_label': data['RO'],
                        'actor_current': {
                            'pos_s': data['actor_traj'][0]['pos_s'],
                            'pos_d': data['actor_traj'][0]['pos_d'],
                            'vel_s': data['actor_traj'][0]['vel_s'],
                            'vel_d': data['actor_traj'][0]['vel_d']
                        }
                    })
                if data['actor_traj'][0]['sensor'] == "lrr":
                    lrr_ls.append({
                        'global': data['ego_traj'][0]['global'],
                        'actor_id':data['actor_traj'][0]['id'],
                        'ro_label': data['RO'],
                        'actor_current': {
                            'pos_s': data['actor_traj'][0]['pos_s'],
                            'pos_d': data['actor_traj'][0]['pos_d'],
                            'vel_s': data['actor_traj'][0]['vel_s'],
                            'vel_d': data['actor_traj'][0]['vel_d']
                        }
                    })
                len_data += 1
            
        print(" -- %d samples from json file %s.\n -- %d unmatched data aborted (aborting rate %.4f).\n -- %d data loaded. %d from cam, %d from radar"\
            %(len_json, fname, (len_json-len_data), (1 - len_data/len_json), len_data, len(cam_ls), len(lrr_ls)))
    return cam_ls, lrr_ls

def read_actors_in_displaydata(display_data):
    pass
    
def to_frame_seqs(dataset, seq_len, timestep=0.04):
    """
    reorder samples into seq set
    ro label as T=1.0, F=0.0
    """
    print("\n[Dataset] Converting single samples into continuous frame seq of len %d ..."%seq_len)
    frame_seqs = []
    # abort_count = 0
    for it_begin in range(len(dataset)-seq_len):
        frame_seq = []
        id_begin = dataset[it_begin]['actor_id']
        global_begin = dataset[it_begin]['global']
        for i in range(seq_len):
            it = it_begin + i
            if dataset[it]['actor_id'] == id_begin and dataset[it]['global'] == global_begin+timestep*i:
                frame_seq.append(dataset[it])
            else:
                # print("ABORT: Not the same actor OR not continuous frame.")
                break
        if len(frame_seq) == seq_len:
            frame_seqs.append({
                # 'ro': 1.0 if all([frame['ro_label'] for frame in frame_seq]) else 0.0,
                'ro': frame_seq[-1]['ro_label'],
                'frame_seq': frame_seq
            })
        # else:
        #     abort_count += 1
        #     print("[global %.2f] ABORT: Not long enough sequence [%d/%d] for actor %d."%(global_begin, i, seq_len, int(id_begin)))
    return frame_seqs

def cvt_fseqs_feavecs(labeled_frame_seqs, expected_x_dim):
    """
    reorder the feature into a 1-dim vector
    ready to feed to NN
    filter 
    """
    print("\n[Dataset] Converting samples into feature vectors ...")
    feavec_ro_pairs = []
    abort_count = 0
    for labeled_frame_seq in labeled_frame_seqs:
        feature_vec = []
        for frame_dic in labeled_frame_seq['frame_seq']:
            feature_vec.append(frame_dic['actor_current']['pos_s'])
            feature_vec.append(frame_dic['actor_current']['pos_d'])
            feature_vec.append(frame_dic['actor_current']['vel_s'])
            feature_vec.append(frame_dic['actor_current']['vel_d'])

        if len(feature_vec)==expected_x_dim:
            feavec_ro_pairs.append({
                'x': np.array(feature_vec),
                'y': labeled_frame_seq['ro']
            })
        else:
            abort_count += 1
    print("%d samples aborted. %d samples saved."%(abort_count, len(feavec_ro_pairs)))
    return feavec_ro_pairs

def data_split(dataset_ls, split_ratio, b_shuffel=False):
    if b_shuffel:
            random.shuffle(dataset_ls)
    i_split = int(len(dataset_ls) * split_ratio)
    data_train = dataset_ls[:i_split]
    data_test = dataset_ls[i_split:]
    print("train/test = (%.2f / %.2f)"%(split_ratio, 1-split_ratio))
    return data_train, data_test

def extend_dataset(dataset, batch_size):
    """
    make the number of samples devidable by batch_size
    """
    num_samples = len(dataset)
    num_extra_samples = batch_size - (num_samples % batch_size)
    extra_indices = np.random.choice(num_samples, size=num_extra_samples, replace=False)
    extra_samples = [dataset[i] for i in extra_indices]
    dataset.extend(extra_samples)
    return dataset

def cvt_nparray(dataset, k_item):
    print("[Dataset] Converting feature vectors into nparray ...")
    timer = Timer()
    timer.start()
    arr = np.array([dataset[0][k_item]])
    for data in dataset[1:]:
        arr = np.concatenate((arr, [data[k_item]]), axis=0)
    timer.stop()
    return arr

def pickle_cache(data, cache_path, pkl_filename):
    file_path = os.path.join(cache_path, pkl_filename)
    print("[Dataset] saving dataset to pkl ...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("[Dataset] Successfully saved dataset to %s."%file_path)

def pickle_load(path):
    print("[Dataset] loading %s..."%path)
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def samples_to_train(dataset, seq_len, split_ratio, batch_size, input_size):
    frame_seqs = to_frame_seqs(dataset, seq_len)
    train_seqs, test_seqs = data_split(frame_seqs, split_ratio, b_shuffel=True)
    train_seqs = extend_dataset(train_seqs, batch_size)
    train_set = cvt_fseqs_feavecs(train_seqs, expected_x_dim=input_size)
    test_set = cvt_fseqs_feavecs(test_seqs, expected_x_dim=input_size)

    train_xy = {
        'x': cvt_nparray(train_set, k_item='x'),
        'y': cvt_nparray(train_set, k_item='y')
    }

    test_xy = {
        'x': cvt_nparray(test_set, k_item='x'),
        'y': cvt_nparray(test_set, k_item='y')
    }
    return train_xy, test_xy


if __name__ == '__main__':
    # Data Path
    # json_folder = "C:\\Users\\SLOFUJ7\\Desktop\\Object_Of_Interest_Detection\\labels"
    json_folder = "data"

    # Config
    seq_len = 10
    split_ratio = 0.8
    batch_size = 64
    input_size = 40

    # Load json as samples
    cam_dataset, lrr_dataset = read_jsons(json_folder)

    # Convert samples to seqs then to feature vectors IN ONE FUNC
    cam_train_set, cam_test_set = samples_to_train(cam_dataset, seq_len, split_ratio, batch_size, input_size)
    print("END")
    