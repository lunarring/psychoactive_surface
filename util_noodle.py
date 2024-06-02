import numpy as np

class NoodleMachine():
    
    def __init__(self):
        self.causes= {}
        self.effect2causes = {}
        self.effect2functs = {}
        self.dict_cause_min = {}
        self.dict_cause_max = {}
    
    def create_noodle(self, cause_names, effect_name, func=np.prod, do_auto_scale=True, init_value=0):
        if type(cause_names) == str:
            cause_names = [cause_names]
        assert hasattr(cause_names, '__getitem__')
        assert type(cause_names[0])==str
        if effect_name in self.effect2causes.keys():
            raise ValueError(f'effect {effect_name} already noodled!')
        self.effect2causes[effect_name] = cause_names
        self.effect2functs[effect_name] = func      
        for cause_name in cause_names:
            if cause_name not in self.causes.keys():
                self.causes[cause_name] = init_value
            if do_auto_scale:
                self.dict_cause_max[cause_name] = None
                self.dict_cause_min[cause_name] = None
            
    def set_cause(self, cause_name, cause_value):
        if cause_name in self.dict_cause_max.keys():
            if self.dict_cause_max[cause_name] is None or cause_value > self.dict_cause_max[cause_name]:
                self.dict_cause_max[cause_name] = cause_value
            if self.dict_cause_min[cause_name] is None or cause_value < self.dict_cause_min[cause_name]:
                self.dict_cause_min[cause_name] = cause_value
            if self.dict_cause_max[cause_name] is not None and self.dict_cause_min[cause_name] is not None:
                if self.dict_cause_max[cause_name] == self.dict_cause_min[cause_name]:
                    self.causes[cause_name] = 0.5
                else:
                    self.causes[cause_name] = (cause_value - self.dict_cause_min[cause_name])/(self.dict_cause_max[cause_name]-self.dict_cause_min[cause_name])
            else:
                self.causes[cause_name] = cause_value
        else:
            self.causes[cause_name] = cause_value
            
    def reset_range(self, cause_name):
        if cause_name in self.dict_cause_max:
            self.dict_cause_max[cause_name] = None
            self.dict_cause_min[cause_name] = None

    def get_effect(self, effect_name):
        if effect_name not in self.effect2causes.keys():
            raise ValueError(f'effect {effect_name} not known')
        cause_names = self.effect2causes[effect_name]
        cause_values = []
        for cause_name in cause_names:
            if cause_name not in self.causes.keys():
                raise ValueError(f'cause {cause_name} not known')
            elif self.causes[cause_name] is None:
                print(f'WARNING: cause {cause_name} not set, returning 0.')
                # raise ValueError(f'cause {cause_name} not set')
            cause_values.append(self.causes[cause_name])
        return self.effect2functs[effect_name](cause_values)