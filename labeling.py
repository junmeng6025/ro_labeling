import argparse
from tqdm import tqdm
from utils import *
from traj_plot import *
from thresholds import ROThresholds
import json
from mat_loader import MatLoader
import pickle

class RO_Discriminator:
    def __init__(self):
        self.th = ROThresholds()

    def r1_keep_in_ego_traj(self, actor_traj):
        '''iterate the actor trajectory list if pos_s at all frames larger than th.s
        and pos_d at all frames smaller than th.d return True else return False
        '''
        pos_s = [frame['pos_s'] for frame in actor_traj]
        pos_d = [frame['pos_d'] for frame in actor_traj]
        if all(p > self.th.s() for p in pos_s):
            if all(p < self.th.d() for p in pos_d):
                return True
        return False

    def r2_cut_in_and_keep_in_ego_traj(self, actor_traj):
        '''iterate the actor trajectory if pos_s at all frames larger than th.s
        and vel_d at the first some frames smaller than th.v_d and pos_d from a certain 
        frame to the end smaller than th.d then return True else return False
        '''
        pos_s = [frame['pos_s'] for frame in actor_traj]
        vel_d = [frame['vel_d'] for frame in actor_traj[:self.th.num_init()]]
        pos_d = [frame['pos_d'] for frame in actor_traj]
        if all(p > self.th.s() for p in pos_s):
            # Check if vel_d is smaller than th.v_d in the first 5 frames
            if all(v < self.th.v_d() for v in vel_d):
                idx = next((i for i, p in enumerate(pos_d) if p < self.th.d()), len(pos_d))
                if all(p < self.th.d() for p in pos_d[idx:]):
                    return True
        return False

    def r3_close_to_ego(self, actor_traj):
        '''iterate the actor trajectory list if pos_s and pos_d at first some frames
        smaller than th.safety_s and th.safety_d than return True else return False
        '''
        pos_s = [frame['pos_s'] for frame in actor_traj[:self.th.num_init()]]
        pos_d = [frame['pos_d'] for frame in actor_traj[:self.th.num_init()]]
        if all(p < self.th.safety_s() for p in pos_s) and all(p < self.th.safety_d() for p in pos_d):
            return True
        return False
    

def discriminate(ego_trajs, actors_trajs, lane_sets):
    """
    discriminate RO according to the RO_rules
    match the data frame, organize as display_data
    """
    label_data = []    # for lable save
    display_data = []  # for dispaly
    ro_count, r1_count, r2_count, r3_count = 0, 0, 0, 0
    discriminator = RO_Discriminator()
    p_bar = tqdm(total=len(actors_trajs), bar_format='{desc:<60}{percentage:3.0f}%|{bar:40}{r_bar}', \
                 desc='[Labeling] Discriminate RO:')
    
    # for display
    frame_len = len(ego_trajs)
    for i in range(frame_len):
        # create lane set for all BV1_LIN channels for a single frame
        lane_set = []
        for lane_chn in lane_sets:
            lane_set.append(lane_chn[i])
        # match the ego_traj, actors_traj, lane_set for every single frame; 
        # -> here actors_traj is initialized as blank, fill in later
        matched_traj = {'frame_id': i, 'global':ego_trajs[i][0]['global'], 'actors_traj': [], 'ego_traj': ego_trajs[i], 'lane_set': lane_set}
        # organize matched data as display_data
        display_data.append(matched_traj)

    # for lable save
    for actor_traj in actors_trajs:
        # ego_iteration = int(actor_traj[0]['global'] / 0.04)
        # ego_traj = ego_trajs[ego_iteration]
        for ego_iteration, ego_traj in enumerate(ego_trajs):
            if ego_traj[0]['global'] == actor_traj[0]['global']:

                ro = False
                r1 = False
                r2 = False
                r3 = False
                if discriminator.r1_keep_in_ego_traj(actor_traj):
                    ro = True
                    r1 = True
                    r1_count += 1
                if discriminator.r2_cut_in_and_keep_in_ego_traj(actor_traj):
                    ro = True
                    r2 = True
                    r2_count += 1
                if discriminator.r3_close_to_ego(actor_traj):
                    ro = True
                    r3 = True
                    r3_count += 1
                if ro:
                    ro_count += 1

                traj_actor_ego = {'RO': True if ro else False, 'actor_traj': actor_traj, 'ego_traj': ego_traj}    
                label_data.append(traj_actor_ego)

                display_actor = {'RO': True if ro else False, 'rules123': (r1, r2, r3), 'actor_traj': actor_traj}  # fill the data in diaplay_data[actors_trajs]
                display_data[ego_iteration]['actors_traj'].append(display_actor)
        
        p_bar.update(1)
    p_bar.close()

    print('[Labeling] RO count: {0}/{1} | r1 count: {2} | r2 count: {3} | r3 count: {4}'.\
          format(ro_count, len(actors_trajs), r1_count, r2_count, r3_count))
    return label_data, display_data

# Utils ========================================================================================
def data_process(mat_loader):
    """
    extract data from sensor signals
    discriminate RO & NRO
    get data for diaplay
    """
    mat_loader.generate()
    ego_trajs = mat_loader.get_ego_paths()
    actor_trajs = mat_loader.get_actors_frenet_paths()
    lane_sets = mat_loader.get_lanes()
    label_data, display_data = discriminate(ego_trajs, actor_trajs, lane_sets)
    return label_data, display_data

def save_labels_as_json(args, label_data, label_filename):
    logfile_path = args.json_folder + 'label_{0}.json'.format(label_filename)
    if not os.path.exists(logfile_path):
        print("[Labeling] Saving label data ...")
        with open(logfile_path, 'w') as f:
            json.dump(label_data, f, indent=4)
            print('[Labeling] Labeled data saved under {}'.format(logfile_path))
    else:
        print("[Labeling] json file already exists, skip ...")

# Save display_data .pkl file
def pickle_cache(data, cache_path, pkl_filename):
    file_path = os.path.join(cache_path, pkl_filename)
    print("[Display] Saving matched traj to pkl ...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("[Display] Successfully saved matched traj to %s."%file_path)


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
                        # = 0: for generate labeling
                        # = 0~len: for load pkl display
    parser.add_argument('--sample_rate', default=1, 
                        help='the sample rate of the actor trajectory which will be processed')
    args = parser.parse_args()
    print('Start with the args: {}'.format(args))

    CACHE_FOLDER = "cache_display"
    SNAPSHOT_PATH = "snapshots"

    mat_loader = MatLoader(args)
    record_name = mat_loader.get_name()

    display_pkl_name = 'display_{0}.pkl'.format(record_name)

    label_data, display_data = data_process(mat_loader)
    save_labels_as_json(args, label_data, record_name)
    pickle_cache(display_data, CACHE_FOLDER, display_pkl_name)
