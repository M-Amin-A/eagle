import os
from tqdm import tqdm
from backdoor_inspection_new import *
import pandas as pd

opt = parse_option()


def save_to_df(df, _anomaly_metric, trigger_type, model_index):
    _raw_dict = {
        'trigger_type': trigger_type,
        'model_index': model_index,
        'anomaly_metric': _anomaly_metric,
    }

    df.loc[len(df)] = _raw_dict
    return df


def _inspect_one_model(model_path):
    print(f'Inspecting model: {model_path}')
    opt.inspect_layer_position = None
    opt.ckpt = model_path
    opt.model = 'preact'
    opt.n_cls = 10
    opt.size = 32
    set_default_settings(opt)

    _anomaly_metric = inspect_saved_model(opt)
    return _anomaly_metric


trigger_types = {
    'warped': (10, 'model.pth.tar'),
    'input_aware': (10, 'model.pth.tar'),
    'clean': (60, 'model.pt'),
    'badnet': (10, 'model.pt'),
    'blended': (10, 'model.pt'),
    'bpp': (10, 'model.pt'),
    'sig': (10, 'model.pt'),
}

for trigger_type in trigger_types.keys():
    # generate empty df
    df = pd.DataFrame(columns=['trigger_type', 'model_index', 'anomaly_metric'])

    num_models, model_name = trigger_types[trigger_type]
    for i in range(1, num_models + 1):
        model_path = f'/kaggle/input/backdoor-attacked-cifar10-classifiers/{trigger_type}/model_{i}/{model_name}'
        _anomaly_metric = _inspect_one_model(model_path)
        df = save_to_df(df, _anomaly_metric, trigger_type, i)

    df.to_csv(f'/kaggle/working/results/{trigger_type}.csv', index=False)