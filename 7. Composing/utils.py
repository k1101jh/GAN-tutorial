import os
import numpy as np


def load_music(data_folder, file_name, n_bars, n_steps_per_bar):
    file = os.path.join(data_folder, file_name)
    
    with np.load(file, encoding='bytes', allow_pickle=True) as f:
        data = f['train']
        
    data_ints = []
    
    for x in data:
        counter = 0
        cont = True
        while cont:
            if not np.any(np.isnan(x[counter:(counter + 4)])):
                cont = False
            else:
                counter += 4
                
        if n_bars * n_steps_per_bar < x.shape[0]:
            data_ints.append(x[counter:(counter + (n_bars * n_steps_per_bar)), :])
            
    data_ints = np.array(data_ints)
    
    n_songs = data_ints.shape[0]
    n_tracks = data_ints.shape[2]
    
    data_ints = data_ints.reshape([n_songs, n_bars, n_steps_per_bar, n_tracks])
    
    max_note = 83
    
    where_are_NaNs = np.isnan(data_ints)
    data_ints[where_are_NaNs] = max_note + 1
    max_note = max_note + 1
    
    data_ints = data_ints.astype(int)
    
    num_classes = max_note + 1
    
    data_binary = np.eye(num_classes)[data_ints]
    data_binary[data_binary == 0] = -1
    data_binary = np.delete(data_binary, max_note, -1)
    
    data_binary = data_binary.transpose([0, 3, 1, 2, 4])
    # batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches
    
    return data_binary, data_ints, data
