import os
from tqdm import tqdm
from backdoor_inspection_new import *
import pandas as pd

opt = parse_option()


def save_to_df(df, _anomaly_metric, trigger_type, model_index, layer_position_index):
    _raw_dict = {
        'trigger_type': trigger_type,
        'model_index': model_index,
        'anomaly_metric': _anomaly_metric,
        'poisoned': trigger_type != 'clean',
        'ldef': layer_position_index,
    }

    df.loc[len(df)] = _raw_dict
    return df


def _inspect_one_model(model_path, layer_position_index):
    print(f'Inspecting model: {model_path}')
    opt.inspect_layer_position = layer_position_index
    opt.ckpt = model_path
    opt.model = 'preact'
    opt.n_cls = 10
    opt.size = 32
    set_default_settings(opt)

    _anomaly_metric = inspect_saved_model(opt)
    return _anomaly_metric


trigger_types = {
    'clean': (60, 'model.pt'),
    'input_aware': (10, 'model.pth.tar'),
    'warped': (10, 'model.pth.tar'),
    'badnet': (10, 'model.pt'),
    'blended': (10, 'model.pt'),
    'bpp': (10, 'model.pt'),
    'sig': (10, 'model.pt'),
}

for ldef in [0, 1, 2, 3, 4, 5]:
    for trigger_type in trigger_types.keys():
        # generate empty df
        df = pd.DataFrame(columns=['trigger_type', 'model_index', 'anomaly_metric', 'poisoned', 'ldef'])

        num_models, model_name = trigger_types[trigger_type]
        for i in range(1, num_models + 1):
            if trigger_type == 'input_aware' and i == 9:
                continue
            model_path = f'/kaggle/input/backdoor-attacked-cifar10-classifiers/{trigger_type}/model_{i}/{model_name}'
            _anomaly_metric = _inspect_one_model(model_path, ldef)
            df = save_to_df(df, _anomaly_metric, trigger_type, i, ldef)

        df.to_csv(f'/kaggle/working/results/{trigger_type}_{ldef}.csv', index=False)
