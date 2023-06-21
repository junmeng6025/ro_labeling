
class ROThresholds:

    def __init__(self):
        # threshold for the distance between actor and ego in frenet coordinate s direction
        self.threshold_s = 0.0
        # threshold for the distance between actor and ego in frenet coordinate d direction
        self.threshold_d = 1.0
        # threshold for the relative velocity in frenet coordinate s direction
        self.threshold_v_s = 0.0
        # threshold for the relative velocity in frenet coordinate d direction
        # actor cut in velocity should smaller than this threshold
        self.threshold_v_d = -0.5

        # threshold for the safety distance in frenet coordinate s direction
        self.threshold_safety_s = 5
        # threshold for the safety distance in frenet coordinate d direction
        self.threshold_safety_d = 1.5

        # number of frames we can have credible initial state
        self.num_init_frame = 5
    
    def recompute_threshold(self, _):
        '''Interface reserved for dynamic threshold computation
        Recompute the threshold based on the different situation
        '''
        pass
    
    def s(self): return self.threshold_s

    def d(self): return self.threshold_d

    def v_s(self): return self.threshold_v_s

    def v_d(self): return self.threshold_v_d

    def safety_s(self): return self.threshold_safety_s

    def safety_d(self): return self.threshold_safety_d

    def num_init(self): return self.num_init_frame

