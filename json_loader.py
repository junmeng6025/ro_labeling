import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import json
import math
import random

EGO_TRAJ_LEN = 125
SEQ_LEN = 10
TIME_STEP = 0.04

# Coord convertion ==================================================================
def get_global_coord(global_p, local_p):
    """
    :param origin:
    :param points:
    :return:
    """

    ox, oy, oyaw = global_p
    px, py  = np.array(local_p[0]), np.array(local_p[1])

    x_global = np.cos(oyaw) * px - np.sin(oyaw) * py + ox
    y_global = np.sin(oyaw) * px + np.cos(oyaw) * py + oy

    return x_global, y_global


def get_relative_coord(origin, points):
    """
    Get pose (x, y, yaw) relative to a given origin, yaw normalized to [-pi,+pi]

    The angle should be given in radians.
    :param points: vector of points
    :param origin: origin point
    :return: vector of points
    """
    ox, oy, oyaw = origin
    px, py, pyaw = np.array(points[0]), np.array(points[1]), np.array(points[2])

    # points coordinates relative to origin
    x_rel = np.cos(oyaw) * (px - ox) + np.sin(oyaw) * (py - oy)
    y_rel = -np.sin(oyaw) * (px - ox) + np.cos(oyaw) * (py - oy)
    yaw_rel = np.arctan2(np.sin(pyaw - oyaw), np.cos(pyaw - oyaw))

    return x_rel, y_rel, yaw_rel


def calc_actor_frenet_single(ego_traj_c, actor_node):
    actor_xy_loc = (actor_node['x'], actor_node['y'])

    # find the closest point on the ego trajectory
    min_dist = 1000000
    min_index = 0
    for i in range(len(ego_traj_c)):
        ego_node =[ego_traj_c[i]['x'], ego_traj_c[i]['y']]
        dist = math.dist(ego_node, actor_xy_loc)
        if dist < min_dist:
            min_dist = dist
            min_index = i
    if ego_traj_c[min_index]['x'] < actor_xy_loc[0]:
        left_index = min_index
        right_index = min_index + 1
    else:
        left_index = min_index - 1
        right_index = min_index
    
    # find the two nodes on the ego trajectory which are closest to the actor trajectory
    left_node = []
    right_node = []
    if min_index == 0:
        left_node = [ego_traj_c[min_index]['x'], ego_traj_c[min_index]['y']]
        right_node = [ego_traj_c[min_index+1]['x'], ego_traj_c[min_index+1]['y']]
    elif min_index == len(ego_traj_c) - 1:
        left_node = [ego_traj_c[min_index-1]['x'], ego_traj_c[min_index-1]['y']]
        right_node = [ego_traj_c[min_index]['x'], ego_traj_c[min_index]['y']]
    else:
        left_node = [ego_traj_c[left_index]['x'], ego_traj_c[left_index]['y']]
        right_node = [ego_traj_c[right_index]['x'], ego_traj_c[right_index]['y']]
    
    # calculate ego travel distance for each ego node from pos_x and pos_y
    ego_travel_dist = []
    ego_travel_dist.append(0)
    for i in range(1, len(ego_traj_c)):
        ego_node = [ego_traj_c[i]['x'], ego_traj_c[i]['y']]
        ego_node_prev = [ego_traj_c[i-1]['x'], ego_traj_c[i-1]['y']]
        ego_travel_dist.append(ego_travel_dist[i-1] + math.dist(ego_node, ego_node_prev))

    # trigonometry to find the frenet coordinate of the actor trajectory
    x_diff_l_r = right_node[0] - left_node[0]
    y_diff_l_r = right_node[1] - left_node[1]
    dist_l_r = math.dist(left_node, right_node)

    x_diff_l_a = left_node[0] - actor_xy_loc[0]
    y_diff_l_a = left_node[1] - actor_xy_loc[1]
    dist_l_a = math.dist(left_node, actor_xy_loc)

    pos_d = abs(x_diff_l_r * y_diff_l_a - y_diff_l_r * x_diff_l_a) / dist_l_r

    ratio = math.sqrt(dist_l_a**2 - pos_d**2) / dist_l_r
    if min_index == len(ego_traj_c) - 1:
        pos_s = ego_travel_dist[min_index]
    else:
        pos_s = ego_travel_dist[min_index-1] + (ego_travel_dist[min_index+1] - ego_travel_dist[min_index-1]) * ratio
    return {'pos_d': pos_d, 'pos_s': pos_s}


# Class JsonLoader =====================================================================
class JsonLoader():
    """
    Load coord from json file and convert to Frenet
    Result:
        Actor's history sequence in Frenet rel to current ego traj of future 125 frames

    """
    def __init__(self):
        # self.folder_path = folder_path
        # self.json_fname = json_fname
        self.ego_whole_path = []
        self.frame_ls_json = []

    # Final calls ----------------------------------------------------------
    def get_samples_on_sensor(self, folder_path, json_fname, sensor):
        self.read_json(folder_path, json_fname, sensor)
        self.extend_glb_pose()
        # frame_ls_cam, frame_ls_lrr = self.split_on_sensors()
        samples = self.gen_actor_history_seq()
        return samples

    def get_ego_whole_path(self):
        return self.ego_whole_path

    # Functional methods ----------------------------------------------------
    # Actor-Ego from json    
    def read_json(self, folder_path, json_fname, sensor, ego_traj_len=EGO_TRAJ_LEN):
        """
        Explaination about coordinations:
        - World:  values directly from sensor signal, recorded by signal
        - Global: relative to ego's satrt pose at frame 0
        - Local:  relative to ego's current pose at current frame

        Output:
        - Actor-EgoTraj pairs
        - does NOT follow time-series. For a changed actor id might trace back
        """
        print("\n[Dataset] Loading json data ...")
        data_ls = json.load(open(os.path.join(folder_path, json_fname), 'r'))

        for i, data in enumerate(data_ls):
            if data['actor_traj'][0]['sensor'] == sensor:
                if data['actor_traj'][0]['global'] == data['ego_traj'][0]['global']:
                    # calc actor pose in world coord
                    origin_ego_pose = (
                        data['ego_traj'][0]['world_x'],
                        data['ego_traj'][0]['world_y'],
                        data['ego_traj'][0]['world_yaw']
                    )
                    rel_actor_position = (
                        data['actor_traj'][0]['pos_x'],
                        data['actor_traj'][0]['pos_y']
                    )
                    wc_x_actor, wc_y_actor = get_global_coord(origin_ego_pose, rel_actor_position)
                    wc_yaw_actor = data['actor_traj'][0]['yaw'] + data['ego_traj'][0]['world_yaw']

                    # write data
                    self.frame_ls_json.append({
                        'global': data['ego_traj'][0]['global'],
                        'actor_id':data['actor_traj'][0]['id'],
                        'ro_label': data['RO'],
                        'sensor': data['actor_traj'][0]['sensor'],
                        'actor_current': {           
                            'global': data['actor_traj'][0]['global'], # global time stamp
                            'loc_x': data['actor_traj'][0]['pos_x'], # local, rel to ego_traj[0] at current frame
                            'loc_y': data['actor_traj'][0]['pos_y'], # local
                            'loc_yaw': data['actor_traj'][0]['yaw'], # local
                            'vel_x': data['actor_traj'][0]['vel_x'],
                            'vel_y': data['actor_traj'][0]['vel_y'],
                            'length': data['actor_traj'][0]['length'],
                            'width': data['actor_traj'][0]['width'],
                            'world_x': wc_x_actor,    # world
                            'world_y': wc_y_actor,    # world
                            'world_yaw': wc_yaw_actor # world
                        },
                        'ego_traj': [
                            {
                                'global': ego_point['global'], # global time stamp
                                'time': ego_point['time'],
                                'loc_x': ego_point['pos_x'],
                                'loc_y': ego_point['pos_y'],
                                'loc_yaw': ego_point['yaw'],
                                'vel_t': ego_point['vel_t'],
                                'world_x': ego_point['world_x'],
                                'world_y': ego_point['world_y'],
                                'world_yaw': ego_point['world_yaw']  # [rad]
                            }
                            for ego_point in data['ego_traj'][:ego_traj_len]
                        ]
                    })


    def extend_glb_pose(self):
        print("\n[Dataset] Extend frame data: ego's and actor's GLOBAL POSE ...")
        ego_start_tuple = (
            self.frame_ls_json[0]['ego_traj'][0]['world_x'],
            self.frame_ls_json[0]['ego_traj'][0]['world_y'],
            self.frame_ls_json[0]['ego_traj'][0]['world_yaw'],
        )

        for frame in self.frame_ls_json:
            # ego traj points
            for ego_point in frame['ego_traj']:
                ego_pose_wc_tuple = (
                    ego_point['world_x'],
                    ego_point['world_y'],
                    ego_point['world_yaw']
                )
                ego_x_rel, ego_y_rel, ego_yaw_rel = get_relative_coord(ego_start_tuple, ego_pose_wc_tuple)
                ego_point['glb_x'] = ego_x_rel
                ego_point['glb_y'] = ego_y_rel
                ego_point['glb_yaw'] = ego_yaw_rel
                    
            # actor current point
            actor_pose_wc_tuple = (
                frame['actor_current']['world_x'],
                frame['actor_current']['world_y'],
                frame['actor_current']['world_yaw']
            )
            actor_x_rel, actor_y_rel, actor_yaw_rel = get_relative_coord(ego_start_tuple, actor_pose_wc_tuple)
            frame['actor_current']['glb_x'] = actor_x_rel
            frame['actor_current']['glb_y'] = actor_y_rel
            frame['actor_current']['glb_yaw'] = actor_yaw_rel
            
            # ego whole path
            self.ego_whole_path.append(
                {
                    'x_glb': frame['ego_traj'][0]['glb_x'],
                    'y_glb': frame['ego_traj'][0]['glb_y'],
                    'yaw_glb': frame['ego_traj'][0]['glb_yaw']
                }
            )


    # def split_on_sensors(self):
    #     frame_ls_cam = [frame for frame in self.frame_ls_json if frame['sensor']=='camera']
    #     frame_ls_lrr = [frame for frame in self.frame_ls_json if frame['sensor']=='lrr']
    #     return frame_ls_cam, frame_ls_lrr
    

    def gen_actor_history_seq(self, seq_len=SEQ_LEN, timestep=TIME_STEP):
        """
        reorder samples into seq set
        ro label as T=1.0, F=0.0

        the last actor, i.e. the actor with the latest global_time, would be seen as the current actor.
            -> actor_current = actor_history_seq[-1]
        all the actors before it would be seen as the history sequence
        """
        print("\n[Dataset] Converting single samples into continuous frame seq of len %d ..."%seq_len)
        print("\n[Dataset] Calc every actor's frenet rel to current ego traj")

        actorseq_egotraj_pairs = []
        for it_begin in range(len(self.frame_ls_json)-seq_len):
            frame_seq = []
            id_begin = self.frame_ls_json[it_begin]['actor_id']
            global_begin = self.frame_ls_json[it_begin]['global']
            for i in range(seq_len):
                it = it_begin + i
                if self.frame_ls_json[it]['actor_id'] == id_begin and self.frame_ls_json[it]['global'] == global_begin+timestep*i:
                    frame_seq.append(self.frame_ls_json[it])
                else:
                    # print("ABORT: Not the same actor OR not continuous frame.")
                    break
            if len(frame_seq) == seq_len:
                ego_traj_cart = [
                    {
                        'x': ego_point['glb_x'],
                        'y': ego_point['glb_y']
                    }
                    for ego_point in frame_seq[-1]['ego_traj']
                ]
                # calc every actor's frenet position rel to current ego traj
                actor_history_seq = [frame['actor_current'] for frame in frame_seq] # index from [0] to [len-1]: from the earliest history to current
                for actor in actor_history_seq:
                    actor_xy_loc = {
                        'x': actor['glb_x'],
                        'y': actor['glb_y']
                    }
                    actor_frenet = calc_actor_frenet_single(ego_traj_cart, actor_xy_loc)
                    # add frenet position to actor_dicts
                    actor['pos_d'] = actor_frenet['pos_d']
                    actor['pos_s'] = actor_frenet['pos_s']

                # calc every actor's frenet velocity
                for j in range(len(actor_history_seq)):
                    # add frenet velocity to actor_dicts
                    if j == len(actor_history_seq)-1: # padding the last node
                        vel_d = (actor_history_seq[j]['pos_d'] - actor_history_seq[j-1]['pos_d']) / timestep
                        vel_s = (actor_history_seq[j]['pos_s'] - actor_history_seq[j-1]['pos_s']) / timestep
                    else:
                        vel_d = (actor_history_seq[j+1]['pos_d'] - actor_history_seq[j]['pos_d']) / timestep
                        vel_s = (actor_history_seq[j+1]['pos_s'] - actor_history_seq[j]['pos_s']) / timestep
                    # add frenet velocity to actor_dicts
                    actor_history_seq[j]['vel_d'] = vel_d
                    actor_history_seq[j]['vel_s'] = vel_s

                # Accumulate ego path nodes
                actorseq_egotraj_pairs.append({
                    'ro': frame_seq[-1]['ro_label'],
                    'global': frame_seq[-1]['global'],
                    'actor_history_seq': actor_history_seq,  # actor_current = actor_history_seq[-1]
                    'ego_traj': frame_seq[-1]['ego_traj'],
                })
            # else:
            #     abort_count += 1
            #     print("[global %.2f] ABORT: Not long enough sequence [%d/%d] for actor %d."%(global_begin, i, seq_len, int(id_begin)))
        #TODO
        """
        Flexibel seq_len:
        this 10 frame is an example, can be 20 frame, can be 5 frame. 
        Also consider, some actor does not have enough history, because they observed recently. 
        """
        return actorseq_egotraj_pairs


# Data Processing ==================================================================
def data_split(dataset_ls, split_ratio, b_shuffel=False):
    if b_shuffel:
            random.shuffle(dataset_ls)
    i_split = int(len(dataset_ls) * split_ratio)
    data_train = dataset_ls[:i_split]
    data_test = dataset_ls[i_split:]
    print("train/test = (%.2f / %.2f)"%(split_ratio, 1-split_ratio))
    return data_train, data_test

def cvt_nparray(dataset, k_item):
    print("[Dataset] Converting feature vectors into nparray ...")
    arr = np.array([dataset[0][k_item]])
    for data in dataset[1:]:
        arr = np.concatenate((arr, [data[k_item]]), axis=0)
    return arr

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

# Class DataProcesser =====================================================================
class DataProcesser():
    def __init__(self, configs):
        # self.sample_ls = sample_ls
        self.keys = ['pos_d', 'pos_s', 'vel_d', 'vel_s']
        self.split_ratio = configs["split_ratio"] # 0.80
        self.batch_size = configs["batch_size"] # 64
        self.expected_x_dim = configs["input_size"] # 4*10

    def samples_to_train(self, sample_ls):
        feavec_ro_pairs = self.gen_feavecs_ro_pairs(sample_ls)
        train_set, test_set = data_split(feavec_ro_pairs, self.split_ratio, b_shuffel=True)
        train_set = extend_dataset(train_set, self.batch_size)

        train_xy = {
            'x': cvt_nparray(train_set, k_item='x'),
            'y': cvt_nparray(train_set, k_item='y')
        }

        test_xy = {
            'x': cvt_nparray(test_set, k_item='x'),
            'y': cvt_nparray(test_set, k_item='y')
        }
        return train_xy, test_xy


    def gen_feavecs_ro_pairs(self, sample_ls):
        """
        reorder the feature into a 1-dim vector
        ready to feed to NN
        filter 
        """
        print("\n[Dataset] Converting samples into feature vectors ...")
        feavec_ro_pairs = []
        abort_count = 0
        for sample in sample_ls:
            """
            sample: ----------------- dependent labeled actor_history_seq - ego_traj pairs
            {
                'ro' ---------------- RO label of current actor
                'global' ------------ global time of current moment,
                'actor_history_seq' - list of actor's history states
                    [
                        0: {}
                        1: {}
                        ...
                        9: {}
                    ]
                'ego_traj' ---------- ego traj from current moment to 5 seconds future
            }
            """
            feature_vec = []
            for actor_dic in sample['actor_history_seq']:
                for k in self.keys:
                    feature_vec.append(actor_dic[k])

            if len(feature_vec)==self.expected_x_dim:
                feavec_ro_pairs.append({
                    'x': np.array(feature_vec),
                    'y': 1.0 if sample['ro'] else 0.0 # bool -> double: T/F -> 1.0/0.0
                })
            else:
                abort_count += 1
        print("%d samples aborted. %d samples saved."%(abort_count, len(feavec_ro_pairs)))
        return feavec_ro_pairs


# Final call
def json_to_dataset(json_folder, json_fname, configs, sensor):
    json_loader = JsonLoader()
    samples = json_loader.get_samples_on_sensor(json_folder, json_fname, sensor)

    data_prosser = DataProcesser(configs)
    train_xy, test_xy = data_prosser.samples_to_train(samples)
    return train_xy, test_xy


if __name__ == "__main__":
    # Debug
    # json_folder = "C:\\Users\\SLOFUJ7\\Desktop\\Object_Of_Interest_Detection\\labels"
    json_folder = "labels"
    json_fname = "label_20210609_123753_BB_split_000.json"

    # Config -- should be loaded from configs.json
    configs = json.load(open('configs.json', 'r'))

    train_set, test_set = json_to_dataset(json_folder, json_fname, configs, sensor="camera")
    print("END")
