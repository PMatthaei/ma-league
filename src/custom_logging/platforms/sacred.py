class CustomSacredLogger:
    def __init__(self, sacred_run_dict):
        self.timestep = None
        self.sacred_info = sacred_run_dict.info

    def log(self, key, value, t, log_type):
        if log_type == "scalar":
            self.log_scalar(key, value, t)
        else:
            pass

    def log_scalar(self, key, value, t):
        if key in self.sacred_info:
            self.sacred_info["{}_T".format(key)].append(t)
            self.sacred_info[key].append(value)
        else:
            self.sacred_info["{}_T".format(key)] = [t]
            self.sacred_info[key] = [value]
