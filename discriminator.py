from tqdm import tqdm
from utils import *
from traj_plot import *
from config import ROThresholds

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
    p_bar = tqdm(total=len(actors_trajs), bar_format='{desc:<50}{percentage:3.0f}%|{bar:20}{r_bar}', \
                 desc='Discriminate RO:')
    
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
        ego_iteration = int(actor_traj[0]['global'] / 0.04)
        ego_traj = ego_trajs[ego_iteration]

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
            traj_actor_ego = {'RO': True, 'actor_traj': actor_traj, 'ego_traj': ego_traj}    
            label_data.append(traj_actor_ego)
            ro_count += 1
            display_actor = {'RO': True, 'rules123': (r1, r2, r3), 'actor_traj': actor_traj}  # fill the data in diaplay_data[actors_trajs]
        else:
            traj_actor_ego = {'RO': False, 'actor_traj': actor_traj, 'ego_traj': ego_traj}
            label_data.append(traj_actor_ego)
            display_actor = {'RO': False, 'rules123': (r1, r2, r3), 'actor_traj': actor_traj}
        display_data[ego_iteration]['actors_traj'].append(display_actor)
        
        p_bar.update(1)
    p_bar.close()

    print('RO count: {0}/{1} | r1 count: {2} | r2 count: {3} | r3 count: {4}'.\
          format(ro_count, len(actors_trajs), r1_count, r2_count, r3_count))
    return label_data, display_data
