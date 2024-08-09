import matplotlib.pyplot as plt
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

for trigger_type in trigger_types:

    arr = pd.read_csv(f'./{trigger_type}.csv')['anomaly_metric'].to_numpy()
    arr = np.clip(arr, 0, 30)

    num_models = len(arr)

    plt.scatter(np.arange(0, num_models, 1) / num_models * 10, arr, label=trigger_type, color = {'green' if trigger_type == 'clean' else 'red'})

plt.legend(loc='upper left')
plt.show()
