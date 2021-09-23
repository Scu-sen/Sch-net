import os
import json
from pathlib import Path
from hashlib import sha1




n_folds = 5
folds = list(range(n_folds))


class audio:
    sampling_rate = 44100
    hop_length = 345 * 2
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    min_seconds = 2

    @classmethod
    def get_config_dict(cls):
        config_dict = dict()
        for key, value in cls.__dict__.items():
            if key[:1] != '_' and \
                    key not in ['get_config_dict', 'get_hash']:
                config_dict[key] = value
        return config_dict

    @classmethod
    def get_hash(cls, **kwargs):
        config_dict = cls.get_config_dict()
        config_dict = {**config_dict, **kwargs}
        hash_str = json.dumps(config_dict,
                              sort_keys=True,
                              ensure_ascii=False,
                              separators=None)
        hash_str = hash_str.encode('utf-8')
        return sha1(hash_str).hexdigest()[:7]



