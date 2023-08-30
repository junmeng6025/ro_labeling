import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keyboard
import lane
from datetime import datetime
import os
import pickle
from mlp_tf import MLP
import numpy as np
import json

# Load .pkl file
def pickle_load(path):
    print("[Display] loading matched traj from pkl ...")
    with open(path, 'rb') as f:
        return pickle.load(f)

# Capture plot
def get_time():
    t = datetime.now()
    t_dict = {
        'yyyy': t.year,
        'mm': t.month,
        'dd': t.day,
        'h': t.hour,
        'min': t.minute,
        's': t.second,
    }
    return t_dict

k = 0
def on_key_capture(event):
    if event.key == 'c':
        global k
        t = get_time()
        save_dir = "snapshots/%04d-%02d-%02d"%(t['yyyy'], t['mm'], t['dd'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fimename = "snap_%02d-%02d-%02d_%04d.png"%(t['h'], t['min'], t['s'], k)
        plt.savefig(os.path.join(save_dir, fimename), dpi=400)
        print("%d: snapshot %04d saved."%(k, k))
        k += 1

# Pause
is_paused = False
def on_key_pause(event):
    global is_paused
    if event.name == 'space':
        is_paused = not is_paused
        msg = "> Plotting paused" if is_paused else "> Plotting continue ..."
        print(msg)

# Exit
is_exit = False
def on_key_exit(event):
    global is_exit
    if event.name == 'esc':
        is_exit = not is_exit
        print("> Exit plotting")

# Reverse
is_rev = False
REV_IT = 100
def on_key_rev(event):
    global is_rev
    if event.name == 'r':
        is_rev = True
        print("> Reverse plotting by %d iterations."%REV_IT)

class Metrics:
    def __init__(self):
        self.tp = 0 # GT=T, Pred=T
        self.fp = 0 # GT=F, Pred=T
        self.tn = 0 # GT=F, Pred=F
        self.fn = 0 # GT=T, Pred=F
        self.recall = 0.0
        self.precision = 0.0
        self.f1 = 0.0
        self.recall_ls = []
        self.precision_ls = []
        self.ap = 0.0

    def tp_count(self):
        self.tp += 1

    def fp_count(self):
        self.fp += 1
    
    def tn_count(self):
        self.tn += 1
    
    def fn_count(self):
        self.fn += 1
    
    def update(self):
        self.recall = self.tp/(self.tp + self.fn) if self.tp + self.fn != 0 else 0.0
        self.recall_ls.append(self.recall)
        self.precision = self.tp/(self.tp + self.fp) if self.tp + self.fp != 0 else 0.0
        self.precision_ls.append(self.precision)
        self.f1 = 2*(self.precision*self.recall)/(self.precision + self.recall) if self.precision + self.recall != 0 else 0.0
        if len(self.precision_ls) != 0 and len(self.recall_ls) != 0:
            self.ap = np.trapz(self.precision_ls, self.recall_ls)

# Traj array Preprocess
def get_points_xy(traj):
    points_x = []
    points_y = []
    for point_dic in traj:
        points_x.append(point_dic['pos_x'])
        points_y.append(point_dic['pos_y'])
    return points_x, points_y

ACTOR_COLOR = {'camera': 'blue', 'lrr':'cyan'}
LANE_TYP = [
    {},                                                 # KEINE
    {'id': 1.0, 'linewidth': 0.75, 'linestyle': '-'},   # durchgezogen
    {'id': 2.0, 'linewidth': 0.5, 'linestyle': '--'},   # gestrichelt
    {'id': 3.0, 'linewidth': 0.5, 'linestyle': ':'},    # Bot_Dots
    {'id': 4.0, 'linewidth': 0.75, 'linestyle': '-'},   # Fahrbanrand
    {'id': 5.0, 'linewidth': 1.0, 'linestyle': '-'},    # Leitplanke
    {'id': 6.0, 'linewidth': 1.0, 'linestyle': '-'},    # Bordstein
    {'id': 7.0, 'linewidth': 1.25, 'linestyle': '-'},   # Mauer
    {'id': 8.0, 'linewidth': 0, 'linestyle': '-'},      # Bitumenfuge
    {'id': 9.0, 'linewidth': 0, 'linestyle': '-'},      # sonstig
    {'id': 10.0, 'linewidth': 0, 'linestyle': '-'},     # N DEF
    {'id': 11.0, 'linewidth': 0, 'linestyle': '-'},     # N DEF
    {'id': 12.0, 'linewidth': 0, 'linestyle': '-'},     # N DEF
    {'id': 13.0, 'linewidth': 0, 'linestyle': '-'},     # N DEF
    {'id': 14.0, 'linewidth': 0, 'linestyle': '-'},     # N DEF
    {'id': 15.0, 'linewidth': 0, 'linestyle': '-'},     # INIT
]

# Display matched ego-actors frames
def matched_traj_plot(display_data, model_path, it=0):
    print("Plotting trajectory ...")
    plt.ion() # to run GUI event loop
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    fig.subplots_adjust(top=0.85)
    keyboard.on_press(on_key_pause)
    keyboard.on_press(on_key_exit)
    # keyboard.on_press(on_key_rev)

    # Pred with MLP:
    mlp = MLP()
    mlp.load_model(model_path)
    input_keys = ['pos_d', 'pos_s', 'vel_d', 'vel_s'] # must be in the same order as in mlp_tf.py
    configs = json.load(open('training_configs.json', 'r'))
    met = Metrics()

    # for it in range(len(display_data)):
    while it < len(display_data):
        # press SPACE to pause
        while is_paused:
            pass
        # press R to reverse
        # while is_rev:
        #     if it > REV_IT:
        #         it -= REV_IT
        #     else:
        #         pass
        #     is_rev = False

        # Display
        display_frame = display_data[it]
        global_time = display_frame['global']
        # Display - get ego traj
        ego_x_ls, ego_y_ls = get_points_xy(display_frame['ego_traj'])
        # Display - process actor traj
        actors_x_arr = []
        actors_y_arr = []
        # actors_history_states = []
        actors_history_x_arr = []
        actors_history_y_arr = []
        ro_ls = []
        ds_ls = []
        objID_ls = []
        lw_ls = []
        rules123_ls = []
        sensor_ls = []
        pred_ls = []

        if len(display_frame['actors_traj']) != 0:
            for actor in display_frame['actors_traj']:
                actor_x, actor_y = get_points_xy(actor['actor_traj'])
                actors_x_arr.append(actor_x)
                actors_y_arr.append(actor_y)
                ro_ls.append(actor['RO'])
                ds_ls.append((actor['actor_traj'][0]['pos_d'], actor['actor_traj'][0]['pos_s']))
                objID_ls.append(actor['actor_traj'][0]['id'])
                lw_ls.append((actor['actor_traj'][0]['length'], actor['actor_traj'][0]['width']))
                rules123_ls.append(actor['rules123'])
                sensor_ls.append(actor['actor_traj'][0]['sensor'])
                
                # Implement MLP pred
                if actor['actor_history'] is not None:
                    pred_input = []
                    for actor_state in actor['actor_history']: # index [0]~[9]: from past to current
                        for k in input_keys:
                            pred_input.append(actor_state[k])
                    pred_result = mlp.pred_single_sample(np.array(pred_input), b_cvt_tf=True, th=configs['conf_th'])
                    actor_history_x, actor_history_y = get_points_xy(actor['actor_history'])
                else:
                    pred_result = None
                    actor_history_x = []
                    actor_history_y = []
                    
                pred_ls.append(pred_result)
                actors_history_x_arr.append(actor_history_x)
                actors_history_y_arr.append(actor_history_y)

        # Display - process lane geometry
        lane_channels = []
        lane_gen = lane.LaneGenerator()
        for lane_dict in display_frame['lane_set']:
            lane_x_arr = []
            lane_y_arr = []
            if lane_dict['AbstandY'] != 4095.0:
                xy_ls = lane_gen.createCartesianGeometry(lane_dict)
                for xy in xy_ls:
                    lane_x_arr.append(xy[0])
                    lane_y_arr.append(xy[1])
                lane_channels.append({'x_arr':lane_x_arr, 'y_arr':lane_y_arr, 'typ': lane_dict['Typ']})

        # Display - plot
        plt.xlim([-5, 300])
        plt.ylim([-10, 10])
        ax.set_xlabel('x(s) [m]', ha='right', x=1.05, fontsize=12)
        ax.set_ylabel('y(d) [m]', ha='right', y=1.1, fontsize=12)
        plt.gca().set_aspect("equal")
        fig.suptitle('iteration: %d / %d;  global time: %.2f [s]'%(it, len(display_data), global_time), fontsize=14, fontweight='bold')
        plt.text(0, -0.50, "RO rules:\n r1: keep in ego traj.\n r2: cut in and keep in ego traj.\n r3: close to ego.",
                 ha='left', va='top', transform=ax.transAxes, fontsize=10)
        plt.text(0, -2, "Press C to capture current plot.\nPress SPACE to pause/continue.", ha='left', va='bottom', transform=ax.transAxes, fontsize=8)
        
        # Plot ego traj
        plt.plot(ego_x_ls, ego_y_ls, label = "ego", color='green')  # marker = '>'
        plt.text(0, 1.05, "ego velocity: %.2f km/h"%(3.6*display_frame['ego_traj'][0]["vel_t"]), ha='left', va='bottom', transform=ax.transAxes, fontsize=10)
        ego_marker = patches.Rectangle((-2.5, -1), 5, 2, linewidth=1, color='orange')
        ego_marker.set_zorder(10)
        ax.add_patch(ego_marker)

        # Plot actor traj
        actors_len = len(display_frame['actors_traj'])
        if actors_len != 0:
            ro_msg = ""
            for i in range(actors_len):
                if ro_ls[i]: # GT=T
                    plt.plot(actors_x_arr[i], actors_y_arr[i], label = "actor %02d"%i,
                            marker = "s", markerfacecolor='none', markeredgecolor="red", markeredgewidth=1)
                    plt.text(actors_x_arr[i][0], actors_y_arr[i][0], '[%d]'%objID_ls[i], fontweight='bold', ha='left', va='bottom', zorder=7)
                    
                    actor_marker = patches.Rectangle((actors_x_arr[i][0]-lw_ls[i][0]/2, actors_y_arr[i][0]-lw_ls[i][1]/2), 
                                                     lw_ls[i][0], lw_ls[i][1], edgecolor='red', linewidth=1, facecolor=ACTOR_COLOR[sensor_ls[i]])
                    if pred_ls[i] is not None:
                        plt.plot(actors_history_x_arr[i], actors_history_y_arr[i], label = "actor history %02d"%i,
                            marker = ".", markerfacecolor='none', markeredgecolor=ACTOR_COLOR[sensor_ls[i]], markeredgewidth=1) # actor_history traj
                        if pred_ls[i]: # Pred=T, GT=T -> tp
                            met.tp_count()
                            met.update()
                            plt.plot(actors_x_arr[i][0], actors_y_arr[i][0], marker = "x", color='red', markersize=5, zorder=6)
                            ro_msg += "\nActor ID [%d]: GT=T; Pred=T; TP. (d: %.2f, s: %.2f);   Rules violation: r1[%s], r2[%s], r3[%s]."\
                                      %(objID_ls[i], ds_ls[i][0], ds_ls[i][1], rules123_ls[i][0],rules123_ls[i][1],rules123_ls[i][2])
                        else: # Pred=F, GT=T -> fn
                            met.fn_count()
                            met.update()
                            plt.plot(actors_x_arr[i][0], actors_y_arr[i][0], marker = "x", color='white', markersize=5, zorder=6)
                            ro_msg += "\nActor ID [%d]: GT=T; Pred=F; FN. (d: %.2f, s: %.2f);   Rules violation: r1[%s], r2[%s], r3[%s]."\
                                      %(objID_ls[i], ds_ls[i][0], ds_ls[i][1], rules123_ls[i][0],rules123_ls[i][1],rules123_ls[i][2])
                    else: # Pred=None
                        ro_msg += "\nActor ID %d: GT=T; Pred=None   (d: %.2f, s: %.2f);   Rules: r1[%s], r2[%s], r3[%s]."\
                                  %(objID_ls[i], ds_ls[i][0], ds_ls[i][1], rules123_ls[i][0],rules123_ls[i][1],rules123_ls[i][2])
                else: # GT=F
                    plt.plot(actors_x_arr[i], actors_y_arr[i], label = "actor %02d"%i, linestyle='--', color=ACTOR_COLOR[sensor_ls[i]])  #  marker = "4"
                    plt.text(actors_x_arr[i][0], actors_y_arr[i][0], '[%d]'%objID_ls[i], ha='left', va='bottom', zorder=7)
                    actor_marker = patches.Rectangle((actors_x_arr[i][0]-lw_ls[i][0]/2, actors_y_arr[i][0]-lw_ls[i][1]/2), 
                                                      lw_ls[i][0], lw_ls[i][1], color=ACTOR_COLOR[sensor_ls[i]])
                    if pred_ls[i] is not None:
                        plt.plot(actors_history_x_arr[i], actors_history_y_arr[i], label = "actor history %02d"%i,
                            marker = ".", markerfacecolor='none', markeredgecolor=ACTOR_COLOR[sensor_ls[i]], markeredgewidth=1) # actor_history traj
                        if pred_ls[i]: # Pred=T, GT=F -> fp
                            met.fp_count()
                            met.update()
                            plt.plot(actors_x_arr[i][0], actors_y_arr[i][0], marker = "x", color='red', markersize=5, zorder=6)
                            ro_msg += "\nActor ID [%d]: GT=F; Pred=T; FP. (d: %.2f, s: %.2f);"\
                                      %(objID_ls[i], ds_ls[i][0], ds_ls[i][1])
                        else: # Pred=F, GT=F -> tn
                            met.tn_count()
                            met.update()
                            plt.plot(actors_x_arr[i][0], actors_y_arr[i][0], marker = "x", color='white', markersize=5, zorder=6)
                            ro_msg += "\nActor ID [%d]: GT=F; Pred=F; TN. (d: %.2f, s: %.2f);"\
                                      %(objID_ls[i], ds_ls[i][0], ds_ls[i][1])
                    else: # Pred=None
                        ro_msg += "\nActor ID %d: GT=F Pred=None   (d: %.2f, s: %.2f);"\
                                  %(objID_ls[i], ds_ls[i][0], ds_ls[i][1])
                ax.add_patch(actor_marker)
                actor_marker.set_zorder(5)
            ro_msg += "\n\nRecall: %.4f; Precision: %.4f; AP: %.4f."%(met.recall, met.precision, met.ap)    
            plt.text(0.25, -0.50, ro_msg, ha='left', va='top', transform=ax.transAxes, fontsize=10)
        
        # Plot lanes
        num_lanes = len(lane_channels)
        if num_lanes != 0:
            for i_lane in range(num_lanes):
                lane_typ = int(lane_channels[i_lane]['typ'])
                plt.plot(lane_channels[i_lane]['x_arr'], lane_channels[i_lane]['y_arr'], 
                         linewidth=LANE_TYP[lane_typ]['linewidth'], 
                         linestyle=LANE_TYP[lane_typ]['linestyle'], 
                         color='black')
                
        #TODO: Plot pred result of 'camera' actors

        fig.canvas.mpl_connect('key_press_event', on_key_capture)
        plt.pause(0.01)
        plt.cla()
        plt.show()
        if is_exit:
            break
        it += 1
    
    plt.close()
    print("Plotting ends.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    parser.add_argument('--start_frame', default=5000,
                        help='the start frame in the recording of the labeling process')
                        # = 0: from beginning
                        # = 0~len: for load pkl display
    parser.add_argument('--pred_on', default=True,
                        help='whether launch RO/NRO prediction')
    parser.add_argument('--model_path', default="saved_models/tf/20230816_161046_ep256_bs64.h5", 
                        help='path to an existing .h5 model file')
    args = parser.parse_args()
    print('Start with the args: {}'.format(args))

    CACHE_PATH = "cache_display"
    record_name = "20210609_123753_BB_split_000"
    display_pkl_name = 'display_{0}.pkl'.format(record_name)
    display_path = os.path.join(CACHE_PATH, display_pkl_name)
    display_data = pickle_load(display_path)

    matched_traj_plot(display_data, model_path=args.model_path, it=args.start_frame)
