import math

class Frenet:
    def __init__(self, ego_traj_c, actor_traj_c):
        self.ego_traj_c = ego_traj_c
        self.actor_traj_c = actor_traj_c

        self.actor_pos_f = self.cart2frenet()
        self.actor_traj_f = self.update_vel_to_frenet()


    def get_trajectory(self):
        return self.actor_traj_f


    def cart2frenet(self):
        traj_frenet = []
        for i in range(len(self.actor_traj_c)):
            traj_frenet.append(self.cart2frenet_single(self.actor_traj_c[i]))
        return traj_frenet
    
    def cart2frenet_single(self, actor_node):
        actor_node = [actor_node['pos_x'], actor_node['pos_y']]

        # find the closest point on the ego trajectory
        min_dist = 1000000
        min_index = 0
        for i in range(len(self.ego_traj_c)):
            ego_node =[self.ego_traj_c[i]['pos_x'], self.ego_traj_c[i]['pos_y']]
            dist = math.dist(ego_node, actor_node)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        if self.ego_traj_c[min_index]['pos_x'] < actor_node[0]:
            left_index = min_index
            right_index = min_index + 1
        else:
            left_index = min_index - 1
            right_index = min_index
        
        # find the two nodes on the ego trajectory which are closest to the actor trajectory
        left_node = []
        right_node = []
        if min_index == 0:
            left_node = [self.ego_traj_c[min_index]['pos_x'], self.ego_traj_c[min_index]['pos_y']]
            right_node = [self.ego_traj_c[min_index+1]['pos_x'], self.ego_traj_c[min_index+1]['pos_y']]
        elif min_index == len(self.ego_traj_c) - 1:
            left_node = [self.ego_traj_c[min_index-1]['pos_x'], self.ego_traj_c[min_index-1]['pos_y']]
            right_node = [self.ego_traj_c[min_index]['pos_x'], self.ego_traj_c[min_index]['pos_y']]
        else:
            left_node = [self.ego_traj_c[left_index]['pos_x'], self.ego_traj_c[left_index]['pos_y']]
            right_node = [self.ego_traj_c[right_index]['pos_x'], self.ego_traj_c[right_index]['pos_y']]
        
        # calculate ego travel distance for each ego node from pos_x and pos_y
        ego_travel_dist = self.get_ego_travel_dist()

        # trigonometry to find the frenet coordinate of the actor trajectory
        frenet = {}
        x_diff_l_r = right_node[0] - left_node[0]
        y_diff_l_r = right_node[1] - left_node[1]
        dist_l_r = math.dist(left_node, right_node)

        x_diff_l_a = left_node[0] - actor_node[0]
        y_diff_l_a = left_node[1] - actor_node[1]
        dist_l_a = math.dist(left_node, actor_node)

        frenet['pos_d'] = abs(x_diff_l_r * y_diff_l_a - y_diff_l_r * x_diff_l_a) / dist_l_r

        ratio = math.sqrt(dist_l_a**2 - frenet['pos_d']**2) / dist_l_r
        if min_index == len(self.ego_traj_c) - 1:
            frenet['pos_s'] = ego_travel_dist[min_index]
        else:
            frenet['pos_s'] = ego_travel_dist[min_index-1] + (ego_travel_dist[min_index+1] - ego_travel_dist[min_index-1]) * ratio

        return frenet
        
    
    def get_ego_travel_dist(self):
        "calculate ego travel distance for each ego node from pos_x and pos_y"
        ego_travel_dist = []
        ego_travel_dist.append(0)
        for i in range(1, len(self.ego_traj_c)):
            ego_node = [self.ego_traj_c[i]['pos_x'], self.ego_traj_c[i]['pos_y']]
            ego_node_prev = [self.ego_traj_c[i-1]['pos_x'], self.ego_traj_c[i-1]['pos_y']]
            ego_travel_dist.append(ego_travel_dist[i-1] + math.dist(ego_node, ego_node_prev))
        return ego_travel_dist
    
    def update_vel_to_frenet(self):
        "calculate frenet velocity for each actor node"
        frenet = []
        for i in range(len(self.actor_pos_f)):
            if i == 0: # padding the first node
                vel_s = (self.actor_pos_f[1]['pos_s'] - self.actor_pos_f[0]['pos_s']) / 0.04
                vel_d = (self.actor_pos_f[1]['pos_d'] - self.actor_pos_f[0]['pos_d']) / 0.04
            else:
                vel_s = (self.actor_pos_f[i]['pos_s'] - self.actor_pos_f[i-1]['pos_s']) / 0.04
                vel_d = (self.actor_pos_f[i]['pos_d'] - self.actor_pos_f[i-1]['pos_d']) / 0.04
            frenet.append({
                'pos_s': self.actor_pos_f[i]['pos_s'], 
                'pos_d': self.actor_pos_f[i]['pos_d'],
                'vel_s': vel_s, 
                'vel_d': vel_d
                })
        return frenet



