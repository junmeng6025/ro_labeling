import numpy as np
import pandas as pd
import math


def local2world(position, prev_pos, data, idx_elem):
    """
    Compute absolute positions for EML on X,Y
    :param position:
    :param prev:
    :param data: segment data provided in USK
    :param idx_elem: 0 - EML X, 1 - EML Y
    :return:
    """
    position_world = []
    for idx in range(len(data)):
        cur_pos = data[idx][idx_elem]
        delta = cur_pos - prev_pos
        compensate = 0
        if delta < -8:
            compensate = 16
        elif delta > 8:
            compensate = -16
        position += (delta + compensate)
        position_world.append(position)
        prev_pos = cur_pos 

    return position_world


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
    yaw_rel = np.arctan2(np.sin(pyaw - oyaw), np.cos(pyaw - oyaw))  # limit to plus/minus pi

    x_glb, y_glb = get_global_coord([ox, oy, oyaw], [x_rel, y_rel])
    yaw_glb = yaw_rel + oyaw

    return x_rel, y_rel, yaw_rel, [x_glb, y_glb, yaw_glb]


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


def compute_mock(ego_groundtruth):
    """
    Convert to USK(ego) frame of reference
    1 - EML X, 2 - EML Y, 3 - EML YAW
    :param ego_groundtruth:
    :return:
    """
    # origin coord
    ox, oy, oyaw = ego_groundtruth[0][1], ego_groundtruth[0][2], ego_groundtruth[0][3]
    # get all the points from segment(segment represented by prediction horizon)
    x =   [ego_groundtruth[elem][1] for elem in range(len(ego_groundtruth))]
    y =   [ego_groundtruth[elem][2] for elem in range(len(ego_groundtruth))]
    yaw = [ego_groundtruth[elem][3] for elem in range(len(ego_groundtruth))]

    # compen_xy represent the global coordinates computed from relative coordinates
    x_rel, y_rel, yaw_rel, compen_xyyaw = get_relative_coord([ox, oy, oyaw], [x, y, yaw])

    mp_mock = ego_groundtruth
    for idx in range(ego_groundtruth.__len__()):
        mp_mock[idx][1] = x_rel[idx]
        mp_mock[idx][2] = y_rel[idx]
        mp_mock[idx][3] = yaw_rel[idx]
        mp_mock[idx].append(0.0)  # dtrack - covered distance at node; unit [m] - not used at the moment

    return mp_mock, compen_xyyaw

def mock_ego_to_dictlist(mock, compen_xyyaw, global_idx):

    mock_dict = []
    for idx in range(len(mock)):
        mock_dict.append({'global': global_idx * 0.04,
                          'time': mock[idx][0],
                          'pos_x': mock[idx][1],
                          'pos_y': mock[idx][2],
                          'yaw': mock[idx][3],
                          'curv': mock[idx][4],
                          'vel_t': mock[idx][5],
                          'acc_t': mock[idx][6],
                          'distance': mock[idx][7],
                          'world_x': compen_xyyaw[0][idx],
                          'world_y': compen_xyyaw[1][idx],
                          'world_yaw': compen_xyyaw[2][idx]})

    return mock_dict

def array_to_dataframe(array):
    return pd.DataFrame(array, columns=['time', 'id', 'type', 'ref_point', 'width', 'length', 'height', 'pos_x',
                                            'pos_y','vel_x', 'vel_y', 'yaw'])


def set_compensation(obj):

    def set_offset(compen_x, compen_y, type):
        points = [{} for i in range(0, 3)]
        for i in range(0, 3):
            points[i]['pos_x'] = obj['pos_x'] + compen_x[i]
            points[i]['pos_y'] = obj['pos_y'] + compen_y[i]
            points[i]['pos_z'] = obj['pos_z']
            points[i]['height'] = obj['height']
            points[i]['type'] = type[i]
        
        return points
        
    if obj['ref_point'] == 1.0: # front left
        spherical = set_offset([0.0, 0.0, obj['length']], [0.0, obj['width'], 0.0], 
                                ['front_left', 'front_right', 'rear_left'])
    elif obj['ref_point'] == 14.0: #front left upper point
        spherical = set_offset([0.0, 0.0, obj['length']], [0.0, obj['width'], 0.0], 
                                ['front_left', 'front_right', 'rear_left'])
    elif obj['ref_point'] == 2.0: # front center
        spherical = set_offset([0.0, 0.0, obj['length']], 
                                [-0.5 * obj['width'], 0.5 * obj['width'], -0.5 * obj['width']], 
                                ['front_left', 'front_right', 'rear_left'])
    elif obj['ref_point'] == 3.0: # front right
        spherical = set_offset([0.0, 0.0, obj['length']], [0.0, -obj['width'], 0.0], 
                                ['front_right', 'front_left', 'rear_right'])
    elif obj['ref_point'] == 15.0: # front right upper point
        spherical = set_offset([0.0, 0.0, obj['length']], [0.0, -obj['width'], 0.0], 
                                ['front_right', 'front_left', 'rear_right'])
    elif obj['ref_point'] == 4.0: # middle right
        spherical = set_offset([-0.5 * obj['length'], -0.5 * obj['length'], 0.5 * obj['length']], 
                                [0.0, -obj['width'], 0.0], 
                                ['front_right', 'front_left', 'rear_right'])
    elif obj['ref_point'] == 5.0: # rear right
        spherical = set_offset([0.0, 0.0, obj['length']], [0.5, obj['width'], 0.0], 
                                ['rear_right', 'rear_left', 'front_right'])
    elif obj['ref_point'] == 12.0: # rear right upper point
        spherical = set_offset([0.0, 0.0, obj['length']], [0.5, obj['width'], 0.0], 
                                ['rear_right', 'rear_left', 'front_right'])
    elif obj['ref_point'] == 11.0: # arbitrary point corner
        spherical = set_offset([0.0, 0.0, obj['length']], 
                                [-0.5 * obj['width'], 0.5 * obj['width'], -0.5 * obj['width']], 
                                ['rear_right', 'rear_center', 'rear_left'])
    elif obj['ref_point'] == 6.0: # rear center
        spherical = set_offset([0.0, 0.0, obj['length']], 
                               [-0.5 * obj['width'], 0.5 * obj['width'], -0.5 * obj['width']], 
                                ['rear_right', 'rear_left', 'front_right'])
    elif obj['ref_point'] == 10.0: # arbitrary side 
        spherical = set_offset([0.0, 0.0, obj['length']], 
                                [-0.5 * obj['width'], 0.5 * obj['width'], -0.5 * obj['width']], 
                                ['rear_right', 'rear_left', 'front_right'])
    elif obj['ref_point'] == 7.0: # rear left
        spherical = set_offset([0.0, 0.0, obj['length']], [0.0, -obj['width'], 0.0], 
                                ['rear_left', 'rear_right', 'front_left'])
    elif obj['ref_point'] == 13.0: # rear left upper point
        spherical = set_offset([0.0, 0.0, obj['length']], [0.0, -obj['width'], 0.0], 
                                ['rear_left', 'rear_right', 'front_left'])
    elif obj['ref_point'] == 8.0: # middle left
        spherical = set_offset([-0.5 * obj['length'], -0.5 * obj['length'], 0.5 * obj['length']], 
                                [0.0, obj['width'], 0.0], 
                                ['front_left', 'front_right', 'rear_left'])
    elif obj['ref_point'] == 9.0: # middle center
        spherical = set_offset([-0.5 * obj['length'], -0.5 * obj['length'], 0.5 * obj['length']], 
                                [-0.5 * obj['width'], 0.5 * obj['width'], 0.5 * obj['width']], 
                                ['front_left', 'front_right', 'rear_right'])
    else :
        spherical = set_offset([0.0, 0.0, -obj['length']], 
                               [-0.5 * obj['width'], 0.5 * obj['width'], -0.5 * obj['width']], 
                                ['rear_right', 'rear_left', 'front_right'])

    return spherical

def ref_compensation(obj, yaw):

    position = [obj['pos_x'], obj['pos_y']]
    if obj['ref_point'] == 10.0: # some edge
        position = recalculate_ref_point(obj, yaw, 0.0, 0.0)
    elif obj['ref_point'] == 1.0: # front left
        position = recalculate_ref_point(obj, yaw, -0.5, -0.5)
    elif obj['ref_point'] == 2.0: # front middle
        position = recalculate_ref_point(obj, yaw, -0.5, 0.0)
    elif obj['ref_point'] == 3.0: # front right
        position = recalculate_ref_point(obj, yaw, -0.5, 0.5)
    elif obj['ref_point'] == 4.0: # middle right
        position = recalculate_ref_point(obj, yaw, 0.0, 0.5)
    elif obj['ref_point'] == 5.0: # rear right
        pass
    elif obj['ref_point'] == 11.0: # some corner
        position = recalculate_ref_point(obj, yaw, 0.5, 0.5)
    elif obj['ref_point'] == 7.0: # rear left
        position = recalculate_ref_point(obj, yaw, 0.5, -0.5)
    elif obj['ref_point'] == 8.0: # middle left
        position = recalculate_ref_point(obj, yaw, 0.0, -0.5)
    elif obj['ref_point'] == 9.0: # middle rear
        position = recalculate_ref_point(obj, yaw, 0.5, 0.0)
    else:
        pass
    return position

def recalculate_ref_point(obj, yaw, fx, fy):
    position = [obj['pos_x'], obj['pos_y']]
    position[0] = obj['pos_x'] + fx * obj['length'] * math.cos(yaw) - fy * obj['width'] * math.sin(yaw)
    position[1] = obj['pos_y'] + fx * obj['length'] * math.sin(yaw) + fy * obj['width'] * math.cos(yaw)
    return position

def min_element(array, key):
    min = array[0]
    for i in range(0, len(array)):
        if array[i][key] < min[key]:
            min = array[i]
    return min

def polar_to_cartesian(point):
    return [point['distance'] * math.cos(point['azimuth']) * math.sin(point['elevation']),
            point['distance'] * math.sin(point['azimuth']) * math.sin(point['elevation']),]

def points_on_same_side(p1, p2):
    if p1 == 'front_left' and p2 == 'rear_left':
        return True
    elif p1 == 'front_right' and p2 == 'rear_right':
        return True
    elif p1 == 'front_center' and p2 == 'rear_center':
        return True
    else:
        return False
    
def points_on_same_layer(p1, p2):
    if p1 == 'front_left' and p2 == 'front_right':
        return True
    elif p1 == 'front_right' and p2 == 'front_center':
        return True
    elif p1 == 'front_left' and p2 == 'front_center':
        return True
    elif p1 == 'rear_right' and p2 == 'rear_left':
        return True
    elif p1 == 'rear_left' and p2 == 'rear_center':
        return True
    elif p1 == 'rear_right' and p2 == 'rear_center':
        return True
    else:
        return False

def sort_points(p1, p2, order):
    if order == 'front_to_rear':
        if p1['type'] == 'front_left' and p2['type'] == 'rear_left':
            return [p1, p2]
        elif p1['type'] == 'front_right' and p2['type'] == 'rear_right':
            return [p1, p2]	
        elif p1['type'] == 'front_center' and p2['type'] == 'rear_center':
            return [p1, p2]
        elif p1['type'] == 'rear_left' and p2['type'] == 'front_left':
            return [p2, p1]
        elif p1['type'] == 'rear_right' and p2['type'] == 'front_right':
            return [p2, p1]
        elif p1['type'] == 'rear_center' and p2['type'] == 'front_center':
            return [p2, p1]
    elif order == 'left_to_right':
        if p1['type'] == 'front_left' and p2['type'] == 'front_right':
            return [p1, p2]
        elif p1['type'] == 'front_right' and p2['type'] == 'front_center':
            return [p2, p1]
        elif p1['type'] == 'front_left' and p2['type'] == 'front_center':
            return [p1, p2]
        elif p1['type'] == 'rear_right' and p2['type'] == 'rear_left':
            return [p2, p1]
        elif p1['type'] == 'rear_left' and p2['type'] == 'rear_center':
            return [p1, p2]
        elif p1['type'] == 'rear_right' and p2['type'] == 'rear_center':
            return [p2, p1]
        elif p1['type'] == 'front_right' and p2['type'] == 'front_left':
            return [p2, p1]
        elif p1['type'] == 'front_center' and p2['type'] == 'front_right':
            return [p1, p2]
        elif p1['type'] == 'front_center' and p2['type'] == 'front_left':
            return [p2, p1]
        elif p1['type'] == 'rear_left' and p2['type'] == 'rear_right':
            return [p1, p2]
        elif p1['type'] == 'rear_center' and p2['type'] == 'rear_left':
            return [p2, p1]
        elif p1['type'] == 'rear_center' and p2['type'] == 'rear_right':
            return [p1, p2]
        

def is_in_ego_range(ego_traj, actor_traj):
    actor_front_x = actor_traj[0]['pos_x']
    actor_back_x = actor_traj[-1]['pos_x']
    ego_front_x = ego_traj[0]['pos_x']
    ego_back_x = ego_traj[-1]['pos_x']
    same_direction_in_range = actor_back_x <= ego_back_x and actor_front_x >= ego_front_x and actor_front_x < actor_back_x
    opposite_direction_in_range = actor_back_x <= ego_back_x and actor_front_x >= ego_front_x and actor_front_x > actor_back_x
    return same_direction_in_range or opposite_direction_in_range

def convolve_smooth(interval, windowsize):
    """
    smooth outliers for LRR actor trajectories
    """
    window = np.ones(int(windowsize)) / float(windowsize)
    return np.convolve(interval, window, 'same')