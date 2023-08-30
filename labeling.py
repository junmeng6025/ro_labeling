import argparse
from tqdm import tqdm
from utils import *
from thresholds import ROThresholds
import json
from mat_loader import MatLoader
import pickle
import glob
import sys
import os
from json_loader import get_global_coord, get_relative_coord, calc_actor_frenet_single, get_current_actor_rel_previous

SEQ_LEN = 10
TIME_STEP = 0.04

BV2_CONF = (1.892, -0.023, 1.513) # Position x, y, z

class RO_Discriminator:
    def __init__(self):
        self.th = ROThresholds()

    def r1_keep_in_ego_traj(self, actor_traj):
        '''
        iterate the actor trajectory list if pos_s at all frames larger than th.s
        and pos_d at all frames smaller than th.d return True else return False
        '''
        pos_s = [frame['pos_s'] for frame in actor_traj]
        pos_d = [frame['pos_d'] for frame in actor_traj]
        if all(s > self.th.s() for s in pos_s):
            if all(d < self.th.d() for d in pos_d):
                return True
        return False

    def r2_cut_in_and_keep_in_ego_traj(self, actor_traj):
        '''
        iterate the actor trajectory if pos_s at all frames larger than th.s
        and vel_d at the first some frames smaller than th.v_d and pos_d from a certain 
        frame to the end smaller than th.d then return True else return False
        '''
        pos_s = [frame['pos_s'] for frame in actor_traj]
        vel_d = [frame['vel_d'] for frame in actor_traj[:self.th.num_init()]]
        pos_d = [frame['pos_d'] for frame in actor_traj]
        if all(s > self.th.s() for s in pos_s):
            # Check if vel_d is smaller than th.v_d in the first 5 frames
            if all(vd < self.th.v_d() for vd in vel_d):
                idx = next((i for i, p in enumerate(pos_d) if p < self.th.d()), len(pos_d))
                if all(d < self.th.d() for d in pos_d[idx:]):
                    return True
        return False

    def r3_close_to_ego(self, actor_traj):
        '''
        iterate the actor trajectory list if pos_s and pos_d at first some frames
        smaller than th.safety_s and th.safety_d than return True else return False
        '''
        pos_s = [frame['pos_s'] for frame in actor_traj[:self.th.num_init()]]
        pos_d = [frame['pos_d'] for frame in actor_traj[:self.th.num_init()]]
        if all(s < self.th.safety_s() for s in pos_s) and all(d < self.th.safety_d() for d in pos_d):
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
    
    # generate label data & display data
    actor_history_ls = []
    id_init = actors_trajs[0][0]['id']
    global_init = round(actors_trajs[0][0]['global'], 2)
    ego_recording_start = (
        ego_trajs[0][0]['world_x'],
        ego_trajs[0][0]['world_y'],
        ego_trajs[0][0]['world_yaw']
    ) # ego_traj[0] of frame 0
    label_data.append({'ego_recording_start': ego_recording_start})
    # MOD: add sensor config
    # label_data.append({
    #     'ego_recording_start': ego_recording_start,
    #     'BV2_config': (x, y, z),
    #     'LRR_config': (x, y, z)})

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

    for actor_traj in actors_trajs:
        if actor_traj[0]['id'] == id_init and round(actor_traj[0]['global'], 2) == round((global_init + TIME_STEP*len(actor_history_ls)), 2):
            # accumulate actor history
            actor_history_ls.append(
                {
                    "global": actor_traj[0]['global'],
                    "actor_id": actor_traj[0]['id'],
                    "sensor": actor_traj[0]['sensor'],
                    "local_x": actor_traj[0]['pos_x'], # rel to ego_traj[0] of current frame
                    "local_y": actor_traj[0]['pos_y'],
                    "local_yaw": actor_traj[0]['yaw']
                }
            )
            # print("- DEBUG: Actor id: %d, appeared in %d continuous frames"%(int(id_init), len(actor_history_ls)))
        else: # if not the same actor OR not continuous frame, reset
            id_init = actor_traj[0]['id']
            global_init = round(actor_traj[0]['global'], 2)
            actor_history_ls = []
            actor_history_ls.append(
                {
                    "global": actor_traj[0]['global'],
                    "actor_id": actor_traj[0]['id'],
                    "sensor": actor_traj[0]['sensor'],
                    "local_x": actor_traj[0]['pos_x'],
                    "local_y": actor_traj[0]['pos_y'],
                    "local_yaw": actor_traj[0]['yaw']
                }
            )
            # print("- DEBUG: Reset, new Actor id: %d"%int(id_init))

        for ego_iteration, ego_traj in enumerate(ego_trajs):
            if ego_traj[0]['global'] == actor_traj[0]['global']:
                # ego_iteration = int(actor_traj[0]['global']/TIME_STEP)
                # ego_traj = ego_trajs[ego_iteration]
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

                labeled_actor_ego_pair = {'RO': True if ro else False, 'actor_traj': actor_traj, 'ego_traj': ego_traj}    
                label_data.append(labeled_actor_ego_pair)
                # label_data[0]: ego_recording_origin
                # label_data[1:]: ego-actor pairs

                # write the matched ego to the latest actor history
                actor_history_ls[-1]['ego_traj_wc'] = [
                    {
                        'world_x': ego_point['world_x'],
                        'world_y': ego_point['world_y'],
                        'world_yaw': ego_point['world_yaw']
                    }
                    for ego_point in ego_traj
                ]

                for ego_pose in actor_history_ls[-1]['ego_traj_wc']:
                    ego_x_glb, ego_y_glb, ego_yaw_glb = get_relative_coord(ego_recording_start, (ego_pose['world_x'], ego_pose['world_y'], ego_pose['world_yaw']))
                    ego_pose['glb_x'] = ego_x_glb
                    ego_pose['glb_y'] = ego_y_glb
                    ego_pose['glb_yaw'] = ego_yaw_glb

                ego_pose_current = (
                    ego_traj[0]['world_x'],
                    ego_traj[0]['world_y'],
                    ego_traj[0]['world_yaw'],
                )
                actor_pose_current = (
                    actor_history_ls[-1]['local_x'],
                    actor_history_ls[-1]['local_y'],
                    actor_history_ls[-1]['local_yaw'],
                )

                actor_pose_glb = get_current_actor_rel_previous(ego_i=ego_recording_start, actor_n=actor_pose_current, ego_n=ego_pose_current)
                actor_history_ls[-1]['glb_x'] = actor_pose_glb['x_a_i']
                actor_history_ls[-1]['glb_y'] = actor_pose_glb['y_a_i']
                actor_history_ls[-1]['glb_yaw'] = actor_pose_glb['yaw_a_i']

                # 
                actor_history = deep_copy_custom(actor_history_ls[-SEQ_LEN:]) if len(actor_history_ls) >= SEQ_LEN else None
                if actor_history is not None:  # extend history actors' frenet rel to current ego
                    # ego_traj_cart = actor_history[-1]['ego_traj_wc']
                    # for frenet
                    # Frenet calc from GLOBAL -------------------
                    ego_traj_cart = [
                        {
                            'x': ego_cart['glb_x'],
                            'y': ego_cart['glb_y']
                        }
                        for ego_cart in actor_history[-1]['ego_traj_wc'] # the last in history_seq is the current
                    ]

                    # for local
                    ego_current = ( # x, y, yaw
                        actor_history[-1]['ego_traj_wc'][0]['world_x'],
                        actor_history[-1]['ego_traj_wc'][0]['world_y'],
                        actor_history[-1]['ego_traj_wc'][0]['world_yaw']
                    )
                    
                    for actor_history_state in actor_history:
                        # extend history actor's frenet pose rel to current ego_traj
                        # Frenet calc from GLOBAL -------------------
                        actor_node = {
                            'x': actor_history_state['glb_x'],
                            'y': actor_history_state['glb_y']
                        }
                        actor_frenet = calc_actor_frenet_single(ego_traj_cart, actor_node)
                        actor_history_state['pos_s'] = actor_frenet['pos_s']
                        actor_history_state['pos_d'] = actor_frenet['pos_d']

                        # calc history actor's local pose rel to current ego origin (i.e. ego_traj[0])
                        # actor_rel_ego_current = get_relative_coord(ego_current_global, actor_pose_global)
                        actor_rel_correspond_ego = ( # x, y, yaw
                            actor_history_state['local_x'],
                            actor_history_state['local_y'],
                            actor_history_state['local_yaw']
                        )
                        ego_correspond = ( # x, y, yaw
                            actor_history_state['ego_traj_wc'][0]['world_x'],
                            actor_history_state['ego_traj_wc'][0]['world_y'],
                            actor_history_state['ego_traj_wc'][0]['world_yaw']
                        )

                        actor_rel_current_ego = get_current_actor_rel_previous(
                            ego_i=ego_current, 
                            actor_n=actor_rel_correspond_ego, 
                            ego_n=ego_correspond)
                        actor_history_state['pos_x'] = actor_rel_current_ego['x_a_i'] # rel to current ego origin
                        actor_history_state['pos_y'] = actor_rel_current_ego['y_a_i'] # rel to current ego origin
                        
                    for j in range(len(actor_history)):
                        # add frenet velocity to actor_dicts
                        if j == len(actor_history)-1: # padding the latest node
                            vel_d = (actor_history[j]['pos_d'] - actor_history[j-1]['pos_d']) / TIME_STEP
                            vel_s = (actor_history[j]['pos_s'] - actor_history[j-1]['pos_s']) / TIME_STEP
                        else:
                            vel_d = (actor_history[j+1]['pos_d'] - actor_history[j]['pos_d']) / TIME_STEP
                            vel_s = (actor_history[j+1]['pos_s'] - actor_history[j]['pos_s']) / TIME_STEP
                        # add frenet velocity to actor_dicts
                        actor_history[j]['vel_d'] = vel_d
                        actor_history[j]['vel_s'] = vel_s

                display_actor = {'RO': True if ro else False, 'rules123': (r1, r2, r3), 'actor_traj': actor_traj, 'actor_history': actor_history}  # write the data in diaplay_data['actors_trajs']
                display_data[ego_iteration]['actors_traj'].append(display_actor)
        
        p_bar.update(1)
    p_bar.close()

    print('[Labeling] RO count: {0}/{1} | r1 count: {2} | r2 count: {3} | r3 count: {4}'.\
          format(ro_count, len(actors_trajs), r1_count, r2_count, r3_count))
    return label_data, display_data

# Utils ========================================================================================
# Custom deep copy
def deep_copy_custom(obj):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, list):
        return [deep_copy_custom(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deep_copy_custom(value) for key, value in obj.items()}
    
# Save data to local
def save_labels_as_json(args, label_data, record_name):
    """
    Save label_data to ./labels
    """
    lable_path = args.label_folder + 'label_{0}.json'.format(record_name)
    if not os.path.exists(lable_path):
        print("[Labeling] Saving label data ...")
        with open(lable_path, 'w') as f:
            json.dump(label_data, f, indent=4)
            print('[Labeling] Labeled data saved under {}'.format(lable_path))
    else:
        print("[Labeling] json file already exists, skip ...")

def save_display_as_pkl(args, display_data, record_name):
    """
    Save display_data to ./cache_display
    """
    display_path = args.display_folder + 'display_{0}.pkl'.format(record_name)
    print("[Display] Saving matched traj to pkl ...")
    with open(display_path, 'wb') as f:
        pickle.dump(display_data, f)
        print("[Display] Successfully saved matched traj to %s."%display_path)

def pickle_load(path):
    print("[MatLoader] loading processed mat from pkl ...")
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def pickle_cache(data, cache_path, pkl_filename):
    file_path = os.path.join(cache_path, pkl_filename)
    print("[MatLoader] Saving preprocessed mat to pkl ...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("[MatLoader] Successfully saved preprocessed mat to %s."%file_path)

# Final call
def mat_process(args, filepath):
    """
    extract data from sensor signals
    discriminate RO & NRO -> save as .json
    get data for diaplay -> save as .pkl
    """
    mat_loader = MatLoader(args, filepath)
    record_name = mat_loader.get_name()
    mat_loader.generate()
    ego_trajs = mat_loader.get_ego_paths()
    actor_trajs = mat_loader.get_actors_frenet_paths()
    lane_sets = mat_loader.get_lanes()
    return ego_trajs, actor_trajs, lane_sets, record_name

    # DEBUG ===================================================
    # if args.if_save_preprocess_mat:
    #     mat_cache = {
    #         'ego_trajs': ego_trajs,
    #         'actor_trajs': actor_trajs,
    #         'lane_sets': lane_sets
    #     }
    #     pickle_cache(mat_cache, 'debug', 'mat_%s.pkl'%record_name)
    # # =========================================================

    # label_data, display_data = discriminate(ego_trajs, actor_trajs, lane_sets)
    # save_labels_as_json(args, label_data, record_name)
    # save_display_as_pkl(args, display_data, record_name)
    # return record_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--mat_folder', default='./mat_data/', 
                        help='folder containing .mat recording')
    parser.add_argument('--if_save_preprocess_mat', default=True, 
                        help='save preprocessed recording as .pkl to local?')
    parser.add_argument('--if_load_preprocess_mat', default=False, 
                        help='save preprocessed recording as .pkl to local?')
    parser.add_argument('--record_name', default="local_911_02", 
                        help='if skip generating, record_name MUST be given') # "20210609_123753_BB_split_000"
    parser.add_argument('--label_folder', default='./labels/',
                        help='output path to the folder where labeled training file will be saved.')
    parser.add_argument('--display_folder', default='./cache_display/',
                        help='output path to the folder where matched traj file will be saved.')
    parser.add_argument('--range', default=12.0,
                        help='range of the ego trajectory which will be processed [s].')
    parser.add_argument('--gen_start_frame', default=0,
                        help='the start frame in the recording of the labeling process')
                        # = 0: for generate labeling
                        # = 0~len: for load pkl display
    parser.add_argument('--sample_rate', default=1, 
                        help='the sample rate of the actor trajectory which will be processed')
    args = parser.parse_args()
    print('Start with the args: {}'.format(args))

    # DEBUG ===================================================
    record_name = args.record_name
    if args.if_load_preprocess_mat and os.path.exists("debug/mat_%s.pkl"%record_name):
        
        cache_mat = pickle_load("debug/mat_%s.pkl"%record_name)
        label_data, display_data = discriminate(cache_mat['ego_trajs'], cache_mat['actor_trajs'], cache_mat['lane_sets'])
        save_labels_as_json(args, label_data, record_name)
        save_display_as_pkl(args, display_data, record_name)
    # =========================================================
    else:
        filepath_ls = glob.glob("{x}/*.{y}".format(x=args.mat_folder, y='mat'))
        dataname_ls = [filepath.split('\\')[-1].split('.')[0]  for filepath in filepath_ls]
        if len(filepath_ls):
            print("[Mat loader] %d .mat files found."%len(filepath_ls))
            for i, data_name in enumerate(dataname_ls):
                print("\n -[%d] %s"%(i+1, data_name))
            mat_idx = int(input("\n[Mat loader] Select one mat file to load. (input 0 to load all) >>> "))
            filepath = filepath_ls[mat_idx-1]
            ego_trajs, actor_trajs, lane_sets, record_name = mat_process(args, filepath)
                
            if args.if_save_preprocess_mat:
                mat_cache = {
                    'ego_trajs': ego_trajs,
                    'actor_trajs': actor_trajs,
                    'lane_sets': lane_sets
                }
                pickle_cache(mat_cache, 'debug', 'mat_%s.pkl'%record_name)
            
            # label_data, display_data = discriminate(ego_trajs, actor_trajs, lane_sets)
            # save_labels_as_json(args, label_data, record_name)
            # save_display_as_pkl(args, display_data, record_name)
        else:
            sys.exit('[Mat loader] .mat not found')
