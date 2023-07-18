import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import pickle
from mat_loader import MatLoader
import json
import math

# def ego_traj_gen(mat_data_folder, mat_fname):
#     data_loader = MatLoader()
#     data_loader.generate_ego_paths()
#     ego_traj_wc = data_loader.get_ego_traj_wc() # FOR DEBUG
#     pickle_cache(ego_traj_wc, "debug", "traj_%s.pkl%"%mat_fname) # FOR DEBUG

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
    
# Coord convert
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

# Actor-Ego from json    
def read_json(folder_path, json_fname, ego_traj_len=125):
    """
    Explaination about coordinations:
     - World:  values directly from sensor signal, recorded by signal
     - Global: relative to ego's satrt pose at frame 0
     - Local:  relative to ego's current pose at current frame

    Output:
     - Actor-EgoTraj pairs
     - does NOT follow time-series. For a changed actor id might trace back
    """

    data_ls = json.load(open(os.path.join(folder_path, json_fname), 'r'))
    frame_ls = []
    for i, data in enumerate(data_ls):
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
            frame_ls.append({
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
    return frame_ls

def extend_glb_pose(frame_ls):
    print("\n[Dataset] Extend frame data: ego's and actor's GLOBAL POSE ...")
    ego_start_tuple = (
        frame_ls[0]['ego_traj'][0]['world_x'],
        frame_ls[0]['ego_traj'][0]['world_y'],
        frame_ls[0]['ego_traj'][0]['world_yaw'],
    )
    ego_whole_path = []
    for frame in frame_ls:
        # ego traj points
        ego_traj_cart = []
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

            # ego_traj_cart.append(
            #     {
            #         'loc_x': ego_point['loc_x'],
            #         'loc_y': ego_point['loc_y']
            #     }
            # )
                
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
        # # calc actor_current (d, s) frenet
        # actor_xy_loc = {
        #     'loc_x': frame['actor_current']['loc_x'],
        #     'loc_y': frame['actor_current']['loc_y']
        # }
        # actor_frenet = calc_actor_frenet_single(ego_traj_cart, actor_xy_loc)
        # frame['actor_current']['pos_d'] = actor_frenet['pos_d']
        # frame['actor_current']['pos_s'] = actor_frenet['pos_s']
        # ego whole path
        ego_whole_path.append(
            {
                'x_glb': frame['ego_traj'][0]['glb_x'],
                'y_glb': frame['ego_traj'][0]['glb_y'],
                'yaw_glb': frame['ego_traj'][0]['glb_yaw']
            }
        )

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

def split_on_sensors(frame_ls):
    frame_ls_cam = [frame for frame in frame_ls if frame['sensor']=='camera']
    frame_ls_lrr = [frame for frame in frame_ls if frame['sensor']=='lrr']
    return frame_ls_cam, frame_ls_lrr

def get_actor_history_seq(frame_ls, seq_len, timestep=0.04):
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
    for it_begin in range(len(frame_ls)-seq_len):
        frame_seq = []
        id_begin = frame_ls[it_begin]['actor_id']
        global_begin = frame_ls[it_begin]['global']
        for i in range(seq_len):
            it = it_begin + i
            if frame_ls[it]['actor_id'] == id_begin and frame_ls[it]['global'] == global_begin+timestep*i:
                frame_seq.append(frame_ls[it])
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
            # calc every actor's frenet rel to current ego traj
            actor_history_seq = [frame['actor_current'] for frame in frame_seq]
            for actor in actor_history_seq:
                actor_xy_loc = {
                    'x': actor['glb_x'],
                    'y': actor['glb_y']
                }
                actor_frenet = calc_actor_frenet_single(ego_traj_cart, actor_xy_loc)
                # add frenet result to actor_dicts
                actor['pos_d'] = actor_frenet['pos_d']
                actor['pos_s'] = actor_frenet['pos_s']

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
    this 10 frame is an example, can be 20 frame, can be 5 frame. 
    Also consider, some actor does not have enough history, because they observed recently. 
    """
    return actorseq_egotraj_pairs


# Check in plot ================================================================
def plt_egotraj_current_actor(frame_ls):  # check coord cvt
    plt.ion() # to run GUI event loop
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    plt.gca().set_aspect("equal")
    ego_whole_path = []
    pos_ego_origin_tuple = (
        frame_ls[0]['ego_traj'][0]['world_x'],
        frame_ls[0]['ego_traj'][0]['world_y'],
        frame_ls[0]['ego_traj'][0]['world_yaw'],
    )
    for frame in frame_ls:
        # ego traj
        ego_traj = []
        for ego_point in frame['ego_traj']:
            ego_pose_wc_tuple = (
                ego_point['world_x'],
                ego_point['world_y'],
                ego_point['world_yaw']
            )
            ego_x_rel, ego_y_rel, ego_yaw_rel = get_relative_coord(pos_ego_origin_tuple, ego_pose_wc_tuple)
            ego_traj.append(
                {
                    'x_wc': ego_point['world_x'],
                    'y_wc': ego_point['world_y'],
                    'yaw_wc': ego_point['world_yaw'],
                    'x_rel': ego_x_rel,
                    'y_rel': ego_y_rel,
                    'yaw_rel': ego_yaw_rel
                }
            )

        # actor current pose
        actor_pose_wc_tuple = (
            frame['actor_current']['world_x'],
            frame['actor_current']['world_y'],
            frame['actor_current']['world_yaw']
        )
        actor_x_rel, actor_y_rel, actor_yaw_rel = get_relative_coord(pos_ego_origin_tuple, actor_pose_wc_tuple)

        actor_p = {
            'x_wc': frame['actor_current']['world_x'],
            'y_wc': frame['actor_current']['world_y'],
            'yaw_wc': frame['actor_current']['world_yaw'],
            'ro': frame['ro_label'],
            'id': frame['actor_id'],
            'length': frame['actor_current']['length'],
            'width': frame['actor_current']['width'],
            'x_rel': actor_x_rel,
            'y_rel': actor_y_rel,
            'yaw_rel': actor_yaw_rel
        }
        # ego whole path
        ego_whole_path.append(
            {
                'x_glb': ego_traj[0]['x_rel'],
                'y_glb': ego_traj[0]['y_rel'],
                'yaw_glb': ego_traj[0]['yaw_rel']
            }
        )

        plt.plot([etjp['x_rel'] for etjp in ego_traj], [etjp['y_rel'] for etjp in ego_traj], label = "ego_traj", color='green') # ego traj in future 5sec
        plt.plot([eptp['x_glb'] for eptp in ego_whole_path], [eptp['y_glb'] for eptp in ego_whole_path], label = "ego_whole_path", color='black') # ego path history
        plt.plot(actor_p['x_rel'], actor_p['y_rel'], color='red' if actor_p['ro'] else 'blue', marker='s') # current actor position
        plt.plot(ego_traj[0]['x_rel'], ego_traj[0]['y_rel'], color='orange', marker='s') # current ego position

        plt.pause(0.01)
        plt.cla()
        plt.show()

def draw_local_coord(ax, x0, y0, yaw_rad, length=5):
    x1 = x0 + length * math.cos(yaw_rad)
    y1 = y0 + length * math.sin(yaw_rad)
    ax.plot([x0, x1], [y0, y1], color='red', linewidth=1)
    x2 = x0 + length * math.cos(yaw_rad+math.pi)
    y2 = y0 + length * math.cos(yaw_rad+math.pi)
    ax.plot([x0, x2], [y0, y2], color='blue', linewidth=1)

# Ego traj
def cvt_to_origin(xy_dict_ls):
    origin = (xy_dict_ls[0]['x'], xy_dict_ls[0]['y'], xy_dict_ls[0]['yaw'])
    xy_dict_rel_ls = [
        {
            'x': 0,
            'y': 0,
            'yaw': 0
        }
    ]
    for point in xy_dict_ls[1:]:
        point_tuple = (
            point['x'],
            point['y'],
            point['yaw']
        )
        x_rel, y_rel, yaw_rel = get_relative_coord(origin, point_tuple)
        xy_dict_rel_ls.append(
            {
                'x': x_rel,
                'y': y_rel,
                'yaw': yaw_rel
            }
        )
    return xy_dict_rel_ls

def plt_ego_path(xy_dict_ls):
    xy_dict_rel_ls = cvt_to_origin(xy_dict_ls)
    x_ls = []
    y_ls = []
    for xy in xy_dict_rel_ls:
        x_ls.append(xy['x'])
        y_ls.append(xy['y'])
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    plt.gca().set_aspect("equal")
    plt.plot(x_ls, y_ls, label = "ego", color='green')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--data_folder', default='./mat_data/', help='path of the mat file which will be processed')
    parser.add_argument('--logs_folder', default='./labels/',
                        help='output path to the folder where labeled training file will be saved.')
    parser.add_argument('--range', default=4.0,
                        help='range of the actor trajectory which will be processed [s].')
    parser.add_argument('--start_frame', default=0,
                        help='the start frame in the recording of the labeling process')
    parser.add_argument('--sample_rate', default=1, 
                        help='the sample rate of the actor trajectory which will be processed')
    parser.add_argument('--load_pkl', default=True, ##########
                        help='the sample rate of the actor trajectory which will be processed')
    args = parser.parse_args()

    json_folder = 'labels'
    json_fname = 'label_20210609_123753_BB_split_000.json'
    frame_ls = read_json(json_folder, json_fname)
    # plt_egotraj_current_actor(frame_ls)
    extend_glb_pose(frame_ls)
    frame_ls_cam, frame_ls_lrr = split_on_sensors(frame_ls)
    frame_seqs_cam = get_actor_history_seq(frame_ls_cam, 10)
    
    # if not args.load_pkl:
    #     data_loader = MatLoader(args)
    #     record_name = data_loader.get_name()
    #     data_loader.generate_ego_paths()
    #     ego_traj_wc = data_loader.get_ego_traj_wc() # FOR DEBUG
    #     pickle_cache(ego_traj_wc, "debug", "egotraj_%s.pkl"%record_name) # FOR DEBUG
    # else:
    #     ego_traj_wc = pickle_load("debug/egotraj_20210609_123753_BB_split_000.pkl") ##########
    # plt_ego_path(ego_traj_wc)

    print("End")
