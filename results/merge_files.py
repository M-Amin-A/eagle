import numpy as np
import pandas as pd

trigger_types = {
    'badnet': (10, 'model.pt'),
    'blended': (10, 'model.pt'),
    'bpp': (10, 'model.pt'),
    'sig': (10, 'model.pt'),
    'input_aware': (10, 'model.pth.tar'),
    'warped': (10, 'model.pth.tar'),
    'clean': (60, 'model.pt'),
}

merged_df = pd.DataFrame()
for trigger_type in trigger_types.keys():
    new_df = pd.read_csv(f'./{trigger_type}.csv')
    merged_df = pd.concat([merged_df, new_df])

merged_df['poisoned'] = merged_df['trigger_type'] != 'clean'
merged_df['ldef'] = 2
merged_df.to_csv('./res.csv', index = False)