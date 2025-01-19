import numpy as np

txt_file_path = '/content/NeRV_based_EvINR/ECD/slider_depth/events.txt'
npy_file_path = '/content/NeRV_based_EvINR/ECD/slider_depth/events.npy'
events = []
with open(txt_file_path, 'r') as f:
    for line in f:
        t, x, y, p = map(float, line.strip().split())
        events.append([t, x, y, p])
events_array = np.array(events)
np.save(npy_file_path, events_array)
