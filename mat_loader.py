import glob
import sys
import scipy.io as sio
import utils
from frenet import Frenet
import math
import argparse
import random
from tqdm import tqdm

### DEBUG
from debug.radar_filter import save_as_csv

TIME_STEP = 0.04  # Time span of MDF/ADTF signal scan
NR_OF_RAW_MAINPATH_POINTS = 300
NR_OF_CAMERA_ACTORS = 10
NR_OF_LRR_ACTORS = 20
NR_OF_LANE_CHN = 6
NO_ACTOR = 255.0 # No actor is present then the ID from sensor will be 255
FRAGMENT_LENGTH = 4.0  # [s] length of the fragment which will be labeled
RADAR_OBJ_MAX_VEL = 127.875  # [m/s] max velocity of the object in the radar


class MatLoader:
    def __init__(self, args):
        self.args = args
        self.ego_generator = EMLFromMat()
        self.actor_generator = OGMFromMat()
        self.lane_generator = LaneFromMat()
        self.signals = {}
        self.ego_paths = []
        # self.ego_traj_wc = [] # FOR DEBUG
        self.actors_cart_paths = []
        self.actors_frenet_paths = []
        self.lanes = []
        self.high = 0
        self.current = args.start_frame
        self.fpi = int(args.range / TIME_STEP)  # Frame per iteration: number of egoframes will be calculated in one iter
        self.sample_rate = args.sample_rate
        self.data_name = 'default'
        self.load_data()
        
    def generate(self):
        self.generate_ego_paths()
        self.generate_actors_cart_paths()
        self.generate_actors_frenet_paths()
        self.generate_lanes()

    def get_name(self):
        return self.data_name
    
    def get_ego_paths(self):
        return self.ego_paths
    
    def get_actors_cart_paths(self):
        return self.actors_cart_paths
    
    def get_actors_frenet_paths(self):
        return self.actors_frenet_paths
    
    def get_lanes(self):
        return self.lanes
    
    # def get_ego_traj_wc(self):   # FOR DEBUG
    #     return self.ego_traj_wc  # FOR DEBUG
    
    # START fea: load a batch of files ============================================================
    # def find_data(self):
    #     filepath_ls = glob.glob("{x}/*.{y}".format(x=self.args.mat_folder, y='mat'))
    #     if len(filepath_ls):
    #         self.data_name = filepath_ls[0].split('\\')[-1].split('.')[0]
    #         print('Process data: {}'.format(filepath_ls[0]))
    #     else:
    #         sys.exit('No .mat file found')
    # END fea: load a batch of files ==============================================================

    def load_data(self):
        filepath_ls = glob.glob("{x}/*.{y}".format(x=self.args.mat_folder, y='mat'))
        if len(filepath_ls):
            self.data_name = filepath_ls[0].split('\\')[-1].split('.')[0]  # CURRENT: load only one file, cannot handle a batch of files
            print('Process data: {}'.format(filepath_ls[0]))
        else:
            sys.exit('.mat not found')

        mat = sio.loadmat(filepath_ls[0])
        self.flex_ray = mat['FlexRay']
        for k in self.flex_ray.dtype.fields.keys():
            if k == 'Time':
                self.signals['Time'] = self.flex_ray['Time'][0, 0]
                self.high = len(self.signals['Time'])
            else:
                obj = self.flex_ray[k][0, 0]
                for l in obj.dtype.fields.keys():
                    self.signals[l] = obj[l][0, 0]
        for key in list(self.signals.keys()):
            if not self.signals.get(key).any():
                self.signals.pop(key)
        self.signals_length = min([len(self.signals[k]) for k in self.signals.keys()])
        print('[Mat loader] {} frames of signal are loaded'.format(self.signals_length))

    def generate_ego_paths(self):
        for idx in tqdm(range(self.high - self.fpi), bar_format='{desc:<60}{percentage:3.0f}%|{bar:40}{r_bar}', \
                        desc='[Mat loader] Generate Ego trajectories for each frame:'):
            mp_mock, compen_xyyaw = utils.compute_mock(self.make_signal_ego_path())
            # # DEBUG: ego traj in world coord
            # self.ego_traj_wc.append({
            #     'x': compen_xyyaw[0][0],
            #     'y': compen_xyyaw[1][0],
            #     'yaw': compen_xyyaw[2][0]
            # })
            # #---------------------------------
            self.ego_paths.append(utils.mock_ego_to_dictlist(mp_mock, compen_xyyaw, idx))   

    def generate_actors_cart_paths(self):
        for id in tqdm(range(NR_OF_CAMERA_ACTORS), bar_format='{desc:<60}{percentage:3.0f}%|{bar:40}{r_bar}', \
                       desc='[Mat loader] Generate Actors trajectories from BV2:'):
            ### DEBUG
            # actor_path_bv2 = self.make_signal_actor_path(id, 'BV2', b_smooth=False)
            # save_as_csv(actor_path_bv2, csv_folder="debug/bv2_traj", sensor='bv2')
            # self.actors_cart_paths.extend(actor_path_bv2)
            ### ORIGIN
            self.actors_cart_paths.extend(self.make_signal_actor_path(id, 'BV2', b_smooth=False))

        for id in tqdm(range(NR_OF_LRR_ACTORS), bar_format='{desc:<60}{percentage:3.0f}%|{bar:40}{r_bar}', \
                       desc='[Mat loader] Generate Actors trajectories from LRR1:'):
            ### DEBUG
            # actor_path_lrr = self.make_signal_actor_path(id, 'LRR1', b_smooth=True)
            # save_as_csv(actor_path_lrr, csv_folder="debug/lrr_traj", sensor='lrr')
            # self.actors_cart_paths.extend(actor_path_lrr)
            ### ORIGIN
            self.actors_cart_paths.extend(self.make_signal_actor_path(id, 'LRR1', b_smooth=True))
    
    def generate_lanes(self):
        for id_chn in tqdm(range(NR_OF_LANE_CHN), bar_format='{desc:<60}{percentage:3.0f}%|{bar:40}{r_bar}', \
                        desc='[Mat loader] Generate Lane set for each frame:'):
            self.lanes.append(self.make_signal_lane(id_chn))

    def collect_sensor_config(self, sensor):
        found = self.signals.keys()
        conf = []
        for id, _type in sensor:
            if id in found :
                conf.append(self.signals[id][0][0])
            else:
                # signal is present in the ADTF mapping but not in the export file
                # or it has less frames than expected (requested). Either way, we use zero
                conf.append(0)
        return conf

    def collect_ego_signal(self, idx, signals):
        """
        collect EML signals of the frame #idx into eml_vals[]
        :param idx: index of the frame
        :param signals: list of signals rel to ego: 
            - EML_xxx
        :param object_id: id of the actor
        :return: list of paths
        """
        found = self.signals.keys()
        eml_vals = []
        for id, _type in signals:
            if id in found and idx <= len(self.signals[id]):
                eml_vals.append(self.signals[id][idx][0])
            else:
                # signal is present in the ADTF mapping but not in the export file
                # or it has less frames than expected (requested). Either way, we use zero
                eml_vals.append(0)
        return eml_vals
    
    def collect_actor_signal(self, idx, signals, object_id):
        """
        Collects the actor data from the mat file and returns a list of paths
        :param idx: index of the frame
        :param signals: list of signals rel to actors: 
            - BV2_Obj_xxx
            - LRR1_Obj_xxx
        :param object_id: id of the actor
        :return: list of paths
        """
        
        found = self.signals.keys()
        frames = []
        for id, _type in signals[object_id]:
            if id in found and idx < len(self.signals[id]):
                frames.append(self.signals[id][idx][0])
            else:
                # signal is present in the ADTF mapping but not in the export file
                # or it has less frames than expected (requested). Either way, we use zero
                frames.append(0)
        return frames
    
    def collect_lane_signal(self, idx, signals, channel_id):
        """
        Collects the lane parameters of the frame #idx into lane_vals[]
        :param idx: index of the frame
        :param signals: list of signals rel to actors: 
            - BV1_LIN_xxx
        :param channel_id: id of the channel
        :return: list of lane sets
        """
        found = self.signals.keys()
        lane_vals = []
        for id, _type in signals[channel_id]:  # id: BV1_LIN_xxxx
            if id in found and idx < len(self.signals[id]):
                lane_vals.append(self.signals[id][idx][0])  # data saved as 1x1 list, needs index [0] here
            else:
                lane_vals.append(0)
        return lane_vals

    def make_signal_ego_path(self):
        start = self.current
        stop = self.current + self.fpi  # fpi: Frame per iteration -> args.range / TIME_STEP = 300
        stop = min(stop, self.signals_length)
        ego_path_data = []
        for idx in range(start, stop):
            data = self.collect_ego_signal(idx, self.ego_generator.signals)
            ego_path_data.append(data)
        path = self.ego_generator.compute_path(ego_path_data)  # len = 300
        self.current += 1
        return path

    def make_signal_actor_path(self, object_id, sensor, b_smooth): 
        actors_path_data = self.truncate_actor_signal(sensor, object_id)
        actors_trajectory = self.get_trajectory_based_on_front(actors_path_data, b_smooth=b_smooth)
        return actors_trajectory
    
    def make_signal_lane(self, channel_id):
        lane_data = []
        for idx in range(self.high):
            lane_vals = self.collect_lane_signal(idx, self.lane_generator.signals, channel_id)  # collect BeginnX, EndeX ... in BV1_LIN_<CHN>
            lane_data.append(self.lane_generator.mock_lane_dict(lane_vals))  # 300 frames of BV1_LIN_<CHN>
        return lane_data

    def truncate_actor_signal(self, sensor, object_id, length=FRAGMENT_LENGTH):
        """
        Truncates the actor signal to the length of the fragment
        :param object_id: id of the actor
        :param sensor: sensor type
        :return: list of paths
        """
        actors_path_data = []
        mask = self.actor_generator.signals[sensor]
        id_prev = self.collect_actor_signal(0, mask, object_id)[0]
        cursor = 1

        while cursor < self.high - self.fpi:
            actor_path_data = []
            if id_prev == NO_ACTOR:
                data = self.collect_actor_signal(cursor, mask, object_id) # get the next actor
                id_prev = data[0]
            else:
                for idx in range(cursor, self.high - 1):
                    data = self.collect_actor_signal(idx, mask, object_id)
                    if data[0] == id_prev: # same actor
                        id_prev = data[0]
                        data.insert(0, idx - 1)

                        # extract features from the data
                        data = self.extract_features(data, sensor)
                        data['global'] = cursor * TIME_STEP  # BUG: might occur 327.840000000003, which supposed to be 327.84
                        actor_path_data.append(data)
                        # if the length of the fragment is reached then truncate the path
                        if len(actor_path_data) >= length / TIME_STEP: # 4 seconds , 100 frames
                            actors_path_data.append(actor_path_data)
                            break
                    else: # new actor
                        id_prev = data[0] # update id
                        cursor = idx
                        break
            cursor += random.randint(1, self.sample_rate) # random sampling in range of configured sample rate
        return actors_path_data
    
    def extract_features(self, data, sensor='BV2'):
        sensor_config = self.collect_sensor_config(self.actor_generator.conf[sensor])
        if sensor == 'BV2':
            ref_point = self.create_cam_ref_point(data, sensor_config)
            spherical = self.get_spherical_points(ref_point, sensor_config)
            actor = self.calculate_from_points(ref_point, spherical, sensor_config)
        elif sensor == 'LRR1':
            actor = self.get_radar_actor(data, sensor_config)
        return actor

    def create_cam_ref_point(self, data, sensor_config):
        ref_point = {
            'idx': data[0],
            'id': data[1],
            'type': data[2],
            'ref_point': data[3],
            'width': data[4],
            'length': 0.0,
            'height': 0.0,
            'pos_x': data[5] - sensor_config[0],
            'pos_y': data[6] - sensor_config[1],
            'pos_z': - sensor_config[2],
            'vel_x': data[7],
            'vel_y': data[8]
        }
        return ref_point

    def get_spherical_points(self, obj, sensor_config):
        dimensions = self.get_actor_size(obj)
        obj['length'] = dimensions[0]
        obj['width'] = dimensions[1]
        spherical = utils.set_compensation(obj)
        for i in range(0, 3):
            spherical[i]['distance'] = math.sqrt(spherical[i]['pos_x'] ** 2 + spherical[i]['pos_y'] ** 2
                                                 + spherical[i]['pos_z'] ** 2)
            spherical[i]['azimuth'] = math.atan2(spherical[i]['pos_y'], spherical[i]['pos_x'])
            spherical[i]['elevation'] = math.atan2(math.sqrt(spherical[i]['pos_z'] ** 2 + spherical[i]['distance'] ** 2),
                                                   spherical[i]['pos_z'])
            elevation_angle_upper = math.atan2(math.sqrt(spherical[i]['pos_x'] ** 2 + spherical[i]['pos_y'] ** 2), 
                                               spherical[i]['pos_z'] + spherical[i]['height'])
            spherical[i]['delta_elevation'] = spherical[i]['elevation'] - elevation_angle_upper

        return spherical

    def calculate_from_points(self, obj, spherical, sensor_config):
        actor = {}
        actor['time'] = obj['idx'] * TIME_STEP
        actor['id'] = obj['id']
        actor['type'] = obj['type']
        actor['ref_point'] = obj['ref_point']
        dimension = self.get_actor_size(obj)
        actor['width'] = dimension[0]
        actor['length'] = dimension[1]
        actor['height'] = dimension[2]

        sensor_radial = math.sqrt(sensor_config[0] ** 2 + sensor_config[1] ** 2 + sensor_config[2] ** 2)
        sensor_azimuth = math.atan2(sensor_config[1], sensor_config[0])
        eml = self.collect_ego_signal(obj['idx'], self.ego_generator.signals)
        sensor_velocity_x = eml[3] - sensor_radial * math.sin(sensor_azimuth) * eml[2]
        sensor_velocity_y = sensor_radial * math.cos(sensor_azimuth) * eml[2]

        actor['vel_x'] = obj['vel_x'] - sensor_velocity_x
        actor['vel_y'] = obj['vel_y'] - sensor_velocity_y

        ref_point = utils.min_element(spherical, 'distance')
        obj_azimuth = math.atan2(ref_point['pos_y'], ref_point['pos_x'])
        # actor['vel_r'] = actor['vel_x'] * math.cos(obj_azimuth) + actor['vel_y'] * math.sin(obj_azimuth)
        # actor['vel_t'] = actor['vel_y'] * math.cos(obj_azimuth) - actor['vel_x'] * math.sin(obj_azimuth)
        
        actor['yaw'] = self.get_actor_yaw(spherical)
        position = utils.ref_compensation(obj, actor['yaw'])
        actor['pos_x'] = position[0]
        actor['pos_y'] = position[1]
        actor['sensor'] = 'camera'

        return actor
    
    def get_actor_size(self, obj):
        dimension = [0.0, 0.0, 0.0] # Width, Length, Height
        if obj['type'] == 7.0: # passenger car
            dimension = [2.0, 5.0, 1.5]
        elif obj['type'] == 3.0: # pedestrian
            dimension = [1.0, 1.0, 2.0]
        elif obj['type'] == 4.0: # pedestrian group
            dimension = [3.0, 3.0, 2.0]
        elif obj['type'] == 9.0 or obj['type'] == 17.0 or obj['type'] == 18.0: # truck, firefighter, ambulance
            dimension = [2.5, 10.0, 4.0]
        elif obj['type'] == 5.0 or obj['type'] == 6.0 or obj['type'] == 10.0: # bicycle, motorcycle, animal
            dimension = [2.0, 5.0, 1.5]
        elif obj['type'] == 0.0 : # unknown
            dimension = [1.5, 1.5, 1.0]
        else:
            dimension = [2.0, 5.0, 1.5]

        return dimension

    def get_actor_yaw(self, spherical):
        yaw = 0.0
        p1, p2 = [0.0, 0.0], [0.0, 0.0]
        for i in range(0, len(spherical) - 1):
           for j in range(i, len(spherical)):
                if utils.points_on_same_side(spherical[i]['type'], spherical[j]['type']):
                    pts = utils.sort_points(spherical[i], spherical[j], 'front_to_rear')
                    p1 = utils.polar_to_cartesian(pts[0])
                    p2 = utils.polar_to_cartesian(pts[1])
                    yaw = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
                elif utils.points_on_same_layer(spherical[i]['type'], spherical[j]['type']):
                    pts = utils.sort_points(spherical[i], spherical[j], 'left_to_right')
                    p1 = utils.polar_to_cartesian(pts[0])
                    p2 = utils.polar_to_cartesian(pts[1])
                    yaw = math.atan2(p1[1] - p2[1], p1[0] - p2[0]) - math.pi / 2
        
        return yaw
    
    def get_radar_actor(self, data, sensor_config):
        actor = {
            'time': data[0] * TIME_STEP,
            'id': data[1],
            'type': data[2],
            'ref_point': data[3],
            'width': 0.0,
            'length': 0.0,
            'height': 0.0,
            'pos_x': data[6] * math.cos(data[7]) - sensor_config[0],
            'pos_y': data[6] * math.sin(data[7]) - sensor_config[1],
            'azimuth': data[7],
            'yaw': data[8],
            'vel_r': data[9],
            'sensor': 'lrr'
        }
        dimension = self.get_actor_size(actor)
        actor['width'] = dimension[0]
        actor['length'] = dimension[1]
        actor['height'] = dimension[2]
        actor['vel_x'], actor['vel_y'] = self.get_radar_actor_velocity(actor, sensor_config)
        del actor['azimuth'] 
        del actor['vel_r']
        return actor


    def get_radar_actor_velocity(self, actor, sensor_config):
        sensor_radial = math.sqrt(sensor_config[0] ** 2 + sensor_config[1] ** 2)
        sensor_azimuth = math.atan2(sensor_config[1], sensor_config[0])
        sensor_rotation = sensor_config[2] * math.pi / 180
        eml = self.collect_ego_signal(int(actor['time']/TIME_STEP), self.ego_generator.signals)
        sensor_velocity_x = eml[3] - sensor_radial * math.sin(sensor_azimuth) * eml[2]
        sensor_velocity_y = sensor_radial * math.cos(sensor_azimuth) * eml[2]
        yaw_s = actor['azimuth'] + sensor_rotation
        v_sensor_radial = sensor_velocity_x * math.cos(yaw_s) + sensor_velocity_y * math.sin(yaw_s)

        actor_velocity_x, actor_velocity_y = 0.0, 0.0
        epsilon = 0.01
        p_cond = math.cos(actor['yaw'] - actor['azimuth'])
        # If radial velocity is grater than epsilon (grater than 0.01 mps) 
        # and if target vehicle is not moving perpendicular to ego vehicle 
        # (perpendicularity condition grater than epsilon)
        if abs(actor['vel_r']) > epsilon and abs(p_cond) > epsilon:
            actor_velocity = (actor['vel_r'] + v_sensor_radial) / p_cond
            if abs(actor_velocity) > RADAR_OBJ_MAX_VEL:
                # Problem in recomputing radar speed. Target is moving approximatively perpendicular to us, 
                # its absolute speed cannot be computed from radial velocity
                actor_velocity = 0.0
            actor_velocity_x = actor_velocity * math.cos(actor['yaw'] + sensor_rotation)
            actor_velocity_y = actor_velocity * math.sin(actor['yaw'] + sensor_rotation)
        else:
            actor_velocity_x = sensor_velocity_x
            actor_velocity_x = sensor_velocity_y
        return actor_velocity_x, actor_velocity_y

    def get_trajectory_based_on_front(self, actors_path_data, b_smooth):
        for path in actors_path_data:
            idx = int(path[0]['time']/TIME_STEP)
            ego_trajectory = self.ego_paths[idx]
            pos_y_ls = []
            for i in range (0, len(path)):
                path[i]['time'] = ego_trajectory[i]['time']
                # x_global = np.cos(oyaw) * px - np.sin(oyaw) * py + ox
                # y_global = np.sin(oyaw) * px + np.cos(oyaw) * py + oy
                px = path[i]['pos_x']
                py = path[i]['pos_y']
                path[i]['pos_x'] = math.cos(ego_trajectory[i]['yaw'])*px - math.sin(ego_trajectory[i]['yaw'])*py + ego_trajectory[i]['pos_x']
                path[i]['pos_y'] = math.sin(ego_trajectory[i]['yaw'])*px + math.cos(ego_trajectory[i]['yaw'])*py + ego_trajectory[i]['pos_y']

                pos_y_ls.append(path[i]['pos_y'])  # for convolve smooth
                path[i]['vel_x'] = path[i]['vel_x'] + ego_trajectory[i]['vel_t'] * math.cos(ego_trajectory[i]['yaw'])
                path[i]['vel_y'] = path[i]['vel_y'] + ego_trajectory[i]['vel_t'] * math.sin(ego_trajectory[i]['yaw'])
                path[i]['yaw'] = path[i]['yaw'] + ego_trajectory[i]['yaw']

            if b_smooth:
                y_ls_smoothed = utils.convolve_smooth(pos_y_ls, 10)
                for i in range (0, len(path)):
                    path[i]['pos_y'] = y_ls_smoothed[i]  # re-write with smoothed values

        return actors_path_data
    
    def generate_actors_frenet_paths(self):
        pbar = tqdm(total=len(self.actors_cart_paths), bar_format='{desc:<60}{percentage:3.0f}%|{bar:40}{r_bar}', \
                    desc='[Mat loader] Update actor traj to frenet coordinate:')
        for cart_traj in self.actors_cart_paths:
            frame = int(cart_traj[0]['global'] / 0.04)
            ego_traj = self.ego_paths[frame]
            if utils.is_in_ego_range(ego_traj, cart_traj):
                actor_traj = []
                frenet = Frenet(ego_traj, cart_traj)
                frenet_traj = frenet.get_trajectory()
                for idx in range(len(cart_traj)):
                    actor_traj.append(cart_traj[idx] | frenet_traj[idx])
                self.actors_frenet_paths.append(actor_traj)
            pbar.update(1)
        pbar.close()

# Ego traj signals =========================================================
class EMLFromMat:
    def __init__(self):
        """ Specify signals from mat file that computes main path mock """

        self.signals = [
            ('EML_PositionX'      , float), #0
            ('EML_PositionY'      , float), #1
            ('EML_Gierwinkel'     , float), #2
            ('EML_GeschwX'        , float), #3
            ('EML_BeschlX'        , float), #4
            ('EML_BeschlY'        , float), #5
        ]

        self.position_x = 0
        self.position_y = 0
        self.prev_x = 0
        self.prev_y = 0

    def compute_path(self, data):
        """
        Create main path mock, by taking from data a given number of samples
        :param data: EML signals
        :return: main path mock
        """

        # convert XY points to world coordinates, x_wc - x world coord
        x_wc = utils.local2world(self.position_x, self.prev_x, data, 0) # 'EML_PositionX'
        y_wc = utils.local2world(self.position_y, self.prev_y, data, 1) # 'EML_PositionY'

        self.prev_x, self.prev_y = data[0][0], data[0][1]
        self.position_x, self.position_y = x_wc[0], y_wc[0]

        # define a step to get signals from data, based on given number of samples
        samples = NR_OF_RAW_MAINPATH_POINTS
        data_len = len(data)
        if data_len > samples:
            step = int(data_len / samples) + 1
            data_end = min(int(step * samples), data_len)
        else:
            step = 1
            data_end = data_len

        main_path = []
        for data_idx in range(0, data_end, step):
           main_path.append(
               [
                   data_idx * 0.04,           # 'Time'
                   float(x_wc[data_idx]),     # 'EML_PositionX'
                   float(y_wc[data_idx]),     # 'EML_PositionY'
                   float(data[data_idx][2]),  # 'EML_YawAngle'
                   0,                         # 'EML_Kurvature'
                   float(data[data_idx][3]),  # 'EML_VelocityX'
                   float(data[data_idx][4])   # 'EML_AccelerationX'
               ]
           )
        return main_path

# Actor traj signals =========================================================
class OGMFromMat:
    def __init__(self):
        """ Specify signals from mat file that computes OGM mock """

        self.bv2_conf = [
            ('BV2_Sensor_PositionX', float),
            ('BV2_Sensor_PositionY', float),
            ('BV2_Sensor_PositionZ', float)
        ]

        self.bv2_signals = [[
            ('BV2_Obj_{:0>2d}_ID'.format(i), float),
            ('BV2_Obj_{:0>2d}_Klasse'.format(i), float),
            ('BV2_Obj_{:0>2d}_Bezugspunkt'.format(i), float),
            ('BV2_Obj_{:0>2d}_Breite'.format(i), float),
            ('BV2_Obj_{:0>2d}_PositionX'.format(i), float),
            ('BV2_Obj_{:0>2d}_PositionY'.format(i), float),
            ('BV2_Obj_{:0>2d}_GeschwX'.format(i), float),
            ('BV2_Obj_{:0>2d}_GeschwY'.format(i), float),
        ] for i in range(1, 11)]
        
        self.LRR1_conf = [
            ('LRR1_SensorPos_X', float),
            ('LRR1_SensorPos_Y', float),
            ('LRR1_SensorPos_YawStatic', float)
        ]

        self.LRR1_signals = [[
            ('LRR1_Obj_{:0>2d}_ID_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_Klasse_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_Bezugspunkt_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_Breite_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_Laenge_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_RadialDist_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_AzimutWnkl_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_GierWnkl_UF'.format(i), float),
            ('LRR1_Obj_{:0>2d}_RadialGeschw_UF'.format(i), float),
        ] for i in range(1, 21)]

        self.signals = {
            'BV2': self.bv2_signals,
            'LRR1': self.LRR1_signals
        }

        self.conf = {
            'BV2': self.bv2_conf,
            'LRR1': self.LRR1_conf
        }

        self.bv2_ogm = {
            'cartesian':{
                'ref_point': None, 'width': None, 'length': None, 'height': None,
                'pos_x': None, 'pos_y': None, 'pos_z': None
            },
            'spherical':{
                'pos_x': None, 'pos_y': None, 'pos_z': None, 'height': None, 'distance': None,
                'azimuth': None, 'elevation': None, 'delta_elevation': None      
            }}

# Lane signals =========================================================
class LaneFromMat:
    def __init__(self):
        """ Specify signals from mat file that computes lane """
        self.signals = [[
            ('BV1_LIN_{:0>2d}_AbstandY'.format(i), float),
            ('BV1_LIN_{:0>2d}_BeginnX'.format(i), float),
            # ('BV1_LIN_{:0>2d}_Breite'.format(i), float),
            ('BV1_LIN_{:0>2d}_EndeX'.format(i), float),
            # ('BV1_LIN_{:0>2d}_ExistMass'.format(i), float),
            ('BV1_LIN_{:0>2d}_Farbe'.format(i), float),
            ('BV1_LIN_{:0>2d}_GierWnkl'.format(i), float),
            ('BV1_LIN_{:0>2d}_HorKruemm'.format(i), float),
            ('BV1_LIN_{:0>2d}_HorKruemmAend'.format(i), float),
            ('BV1_LIN_{:0>2d}_ID'.format(i), float),
            ('BV1_LIN_{:0>2d}_NachfolgerID'.format(i), float),
            ('BV1_LIN_{:0>2d}_Typ'.format(i), float),
            ('BV1_LIN_{:0>2d}_VorgaengerID'.format(i), float),
        ] for i in range(1, 7)]

        self.keys = [
            'AbstandY',
            'BeginnX',
            'EndeX',
            'Farbe',
            'GierWnkl',
            'HorKreumm',
            'HorKreummAend',
            'ID',
            'NachfolgerID',
            'Typ',
            'VorgaengerID'
        ]

    def mock_lane_dict(self, lane_vals):
        lane_dict = {}
        for i, key in enumerate(self.keys):
            lane_dict[key] = lane_vals[i]
        return lane_dict
