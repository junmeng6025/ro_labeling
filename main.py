import argparse
import json
import os
from labeling import discriminate
from traj_plot import matched_traj_plot
import pickle
from mat_loader import MatLoader
import os

def get_files_in_folder(folder_path):
    files_ls = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            files_ls.append(file_name)
    return files_ls

# Save/Load .pkl file
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
    
def data_process(data_loader):
    """
    extract data from sensor signals
    discriminate RO & NRO
    get data for diaplay
    """
    data_loader.generate()
    ego_trajs = data_loader.get_ego_paths()
    actor_trajs = data_loader.get_actors_frenet_paths()
    lane_sets = data_loader.get_lanes()
    label_data, display_data = discriminate(ego_trajs, actor_trajs, lane_sets)
    return label_data, display_data

def save_labels_as_json(args, label_data, label_filename):
    logfile_path = args.json_folder + 'label_{0}.json'.format(label_filename)
    if not os.path.exists(logfile_path):
        print('Saving label data ...')
        with open(logfile_path, 'w') as f:
            json.dump(label_data, f, indent=4)
            print('Labeled data saved under {}'.format(logfile_path))
    else:
        print("json already exists, skip ...")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--mat_folder', default='./mat_data/', 
                        help='path of the mat file which will be processed')
    parser.add_argument('--json_folder', default='./labels/',
                        help='output path to the folder where labeled training file will be saved.')
    parser.add_argument('--range', default=12.0,
                        help='range of the ego trajectory which will be processed [s].')
    parser.add_argument('--start_frame', default=0,
                        help='the start frame in the recording of the labeling process')
                        # = 0: for first run
                        # = 0~len: for load pkl display
    parser.add_argument('--sample_rate', default=1, 
                        help='the sample rate of the actor trajectory which will be processed')
    parser.add_argument('--load_pkl', default=True, ###########
                        help='decide if load the preprocessed data from .pkl file')
                        # False: for matloader debug
                        # True:  for display
    parser.add_argument('--save_pkl', default=True, 
                        help='decide if save the preprocessed data as .pkl file')
    parser.add_argument('--rec_name', default=None, 
                        help='run a specific recording with a given name (WITHOUT .pkl suffix)')
    args = parser.parse_args()
    print('Start with the args: {}'.format(args))

    # "20210609_123753_BB_split_000"
    CACHE_PATH = "cache_display"
    SNAPSHOT_PATH = "snapshots"

    if args.rec_name is None:
        data_loader = MatLoader(args)
        record_name = data_loader.get_name()
    else:
        assert ('cache_' + args.rec_name + '.pkl') in get_files_in_folder(CACHE_PATH)
        record_name = args.rec_name
    pkl_name = 'cache_{0}.pkl'.format(record_name)
    json_name = 'label_{0}.json'.format(record_name)

    if args.load_pkl:
        cache_path = os.path.join(CACHE_PATH, pkl_name)
        if not os.path.exists(cache_path):
            label_data, display_data = data_process(data_loader)
            save_labels_as_json(args, label_data, record_name)
            pickle_cache(display_data, CACHE_PATH, pkl_name)
        else:
            display_data = pickle_load(cache_path)
            print("Successfully loaded matched traj from %s."%cache_path)
    else:
        label_data, display_data = data_process(data_loader)
        save_labels_as_json(args, label_data, record_name)

    matched_traj_plot(display_data, it=args.start_frame)

    # Load .mat recording
    # Labeling -> .json label file
    # Training -> .h5 model
    # Predict
    # Display: Validate prediction vs. labeling