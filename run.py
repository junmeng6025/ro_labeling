from labeling import *
from mlp_tf import *
from traj_plot_pred import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the labeled training file of related or non-related object from the recording .mat file')
    # Workflow control
    parser.add_argument('--skip_dataset_gen', default=True, # Skip dataset gen?
                        help='Skip generating label and display pkl?')
    parser.add_argument('--record_name', default="20210609_123753_BB_split_000", 
                        help='if skip generating, record_name MUST be given') # "20210609_123753_BB_split_000"
    parser.add_argument('--skip_model_training', default=True, # Skip model training?
                        help='Skip model training?')
    parser.add_argument('--model_path', default="saved_models/tf/20230821_172717_ep200_bs64.h5", 
                        help='if skip model training, model_path MUST be given') # "saved_models/tf/20230816_161046_ep256_bs64.h5"
    # Args for Dataset generation -------------------------------------------------------------------------------------
    parser.add_argument('--mat_folder', default='./mat_data/', 
                        help='folder path of the mat files which will be processed')
    parser.add_argument('--label_folder', default='./labels/',
                        help='output path to the folder where labeled training file will be saved.')
    parser.add_argument('--display_folder', default='./cache_display/',
                        help='output path to the folder where matched traj file will be saved.')
    parser.add_argument('--range', default=12.0,
                        help='range of the ego trajectory which will be processed [s].')
    parser.add_argument('--gen_start_frame', default=0,
                        help='the start frame of the recording when generating dataset')
    parser.add_argument('--sample_rate', default=1, 
                        help='the sample rate of the actor trajectory which will be processed')
    # Args of MLP Training -------------------------------------------------------------------------------------------
    parser.add_argument('--is_dataset_from_pkl', default=False, # For first run set False
                        help='load pre-processed data from pkl')
    # Args of Display -----------------------------------------------------------------------------------------------
    parser.add_argument('--load_start_frame', default=7000,
                        help='the start frame for display')
    
    args = parser.parse_args()
    print('Start with the args: {}'.format(args))

    # Generate Dataset: Label & Display ==================================
    # IN: 
    #   .mat Recording
    # OUT:
    #   .json Labeling
    #   .pkl Display
    if not args.skip_dataset_gen:
        mat_ls = glob.glob("{x}/*.{y}".format(x=args.mat_folder, y='mat'))
        dataname_ls = [filepath.split('\\')[-1].split('.')[0]  for filepath in mat_ls]
        if len(mat_ls):
            print("[Mat loader] %d .mat files found."%len(mat_ls))
            for i, data_name in enumerate(dataname_ls):
                print("\n -[%d] %s"%(i+1, data_name))
            mat_idx = int(input("\n[Mat loader] Select one mat file to load. (input 0 to load all) >>> "))
            filepath = mat_ls[mat_idx-1]
            ego_trajs, actor_trajs, lane_sets, record_name = mat_process(args, filepath)
            
            label_data, display_data = discriminate(ego_trajs, actor_trajs, lane_sets)
            save_labels_as_json(args, label_data, record_name)
            save_display_as_pkl(args, display_data, record_name)
        else:
            sys.exit('[Mat loader] .mat not found')
    else:
        print("[Dataset] Dataset gen skipped ...")
        record_name = args.record_name

    # Train MLP ==========================================================
    # IN: 
    #   .json Labeling
    #   .json Config
    # OUT:
    #   .h5 TensorFlow model
    if not args.skip_model_training:
        fname = record_name
        dataset_path = "cache_dataset/dataset_%s.pkl"%fname
        
        if args.is_dataset_from_pkl and os.path.exists(dataset_path):
            loaded_dict = pickle_load(dataset_path)
            x_train = loaded_dict['x_train']
            y_train = loaded_dict['y_train']
            x_test = loaded_dict['x_test']
            y_test = loaded_dict['y_test']
        else:
            configs = json.load(open('training_configs.json', 'r'))
            cam_train_set, cam_test_set = json_to_dataset(args.label_folder, "label_%s.json"%fname, configs, sensor="camera")
            x_train = cam_train_set['x']
            y_train = cam_train_set['y']
            x_test = cam_test_set['x']
            y_test = cam_test_set['y']

        # Model
        mlp = MLP()
        # if args.mlp_mode == "TRAIN":
        print("\n[Training] Mode: TRAIN\n")
        # Train
        mlp.build_model(configs)
        mlp.train(x_train, y_train)
        # Eval
        mlp.eval(x_test, y_test)
        mlp.disp_history()
        print("[Training] Training finished. Model saved to %s"%mlp.model_path)
        model_path = mlp.model_path
    else:
        print("[Training] Training skipped ...")
        model_path = args.model_path

    # Display ============================================================
    # IN: 
    #   .h5 TensorFlow model
    #   .pkl Display
    # OUT:
    #   None
    display_pkl_name = 'display_{0}.pkl'.format(record_name)
    display_path = os.path.join(args.display_folder, display_pkl_name)
    display_data = pickle_load(display_path)
    matched_traj_plot(display_data, model_path, it=args.load_start_frame)