import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from music21 import note, stream, duration, tempo


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation_func):
        super(Conv3dBlock, self).__init__()
        
        self.layers = nn.Sequential()
        self.layers.add_module("conv3d", nn.Conv3d(in_channels, out_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding))
        
        if activation_func == 'relu':
            self.layers.add_module("activation", nn.ReLU(inplace=True))
        elif activation_func == 'leaky_relu':
            self.layers.add_module("activation", nn.LeakyReLU(0.3, inplace=True))
        
        # nn.init.normal_(self.layers[0].weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        out = self.layers(x)
        
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batch_norm, activation_func):
        super(ConvTransposeBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = nn.Sequential()
        self.layers.add_module("conv_transpose", nn.ConvTranspose2d(in_channels, out_channels,
                                                                    kernel_size=kernel_size,
                                                                    stride=stride,
                                                                    padding=padding))
        if use_batch_norm:
            self.layers.add_module("batch_norm", nn.BatchNorm2d(out_channels))
        
        if activation_func == 'relu':
            self.layers.add_module("activation", nn.ReLU(inplace=True))
        elif activation_func == 'tanh':
            self.layers.add_module("activation", nn.Tanh())
        
        # nn.init.normal_(self.layers[0].weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        return self.layers(x)
        

class TemporalNetwork(nn.Module):
    def __init__(self, n_bars, z_dim):
        super(TemporalNetwork, self).__init__()
        
        self.n_bars = n_bars
        self.z_dim = z_dim
        
        self.conv_transpose_block1 = ConvTransposeBlock(self.z_dim, 1024,
                                                        kernel_size=(2, 1),
                                                        stride=(1, 1),
                                                        padding=(0, 0),
                                                        use_batch_norm=True,
                                                        activation_func='relu')
        self.conv_transpose_block2 = ConvTransposeBlock(1024, self.z_dim,
                                                        kernel_size=(self.n_bars - 1, 1),
                                                        stride=(1, 1),
                                                        padding=(0, 0),
                                                        use_batch_norm=True,
                                                        activation_func='relu')
        
    def forward(self, x):
        x = x.view(x.size(0), self.z_dim, 1, 1)
        out = self.conv_transpose_block1(x)
        out = self.conv_transpose_block2(out)
        out = out.view(x.size(0), self.z_dim, self.n_bars)
        
        return out
        
        
class BarGenerator(nn.Module):
    def __init__(self, z_dim, n_steps_per_bar, n_pitches):
        super(BarGenerator, self).__init__()
        
        self.z_dim = z_dim
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        self.linear = nn.Linear(self.z_dim * 4, 1024)
        self.batch_norm = nn.BatchNorm1d(1024) #, momentum=0.9)
        self.relu= nn.ReLU(inplace=True)
        
        self.num_conv_transpose_layers = 5
        self.channels = [512, 512, 256, 256, 256, 1]
        self.kernel_sizes = [(2, 1), (2, 1), (2, 1), (1, 7), (1, 12)]
        self.strides = [(2, 1), (2, 1), (2, 1), (1, 7), (1, 12)]
        self.paddings = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        self.use_batch_norms = [True, True, True, True, False]
        self.activation_functions = ['relu', 'relu', 'relu', 'relu', 'tanh']
        
        self.layers = self._make_layers()
    
    def _make_layers(self):
        layers = nn.Sequential()
        
        for i in range(self.num_conv_transpose_layers):
            layers.add_module(f'conv_transpose_block_{i + 1}', ConvTransposeBlock(self.channels[i],
                                                                                  self.channels[i + 1],
                                                                                  self.kernel_sizes[i],
                                                                                  self.strides[i],
                                                                                  self.paddings[i],
                                                                                  self.use_batch_norms[i],
                                                                                  self.activation_functions[i]))
        
        return layers
    
    def forward(self, x):
        out = self.linear(x)
        if out.size(0) > 1:
            out = self.batch_norm(out)
        out = self.relu(out)
        out = out.view(out.size(0), 512, 2, 1)
        out = self.layers(out)
        out = out.view(out.size(0), 1, 1, self.n_steps_per_bar, self.n_pitches)
        
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, n_tracks, n_bars, n_steps_per_bar, n_pitches):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        
        # 화음
        self.chord_temp_network = TemporalNetwork(n_bars, z_dim)
        
        # 멜로디
        self.melody_temp_networks = nn.ModuleDict({})
        for track_idx in range(self.n_tracks):
            self.melody_temp_networks.add_module(
                "melodygen_" + str(track_idx),
                TemporalNetwork(n_bars, z_dim)
            )
            # self.melody_temp_networks[track_idx] = TemporalNetwork(n_bars, z_dim)
        # self.melody_temp_networks = [None] * self.n_tracks
        # for track_idx in range(self.n_tracks):
        #     self.melody_temp_networks[track_idx] = TemporalNetwork(n_bars, z_dim)
        
        # 트랙마다 마디 생성자를 만듭니다
        self.bar_generators = nn.ModuleDict({})
        for track_idx in range(self.n_tracks):
            self.bar_generators.add_module(
                "bargen_" + str(track_idx),
                BarGenerator(z_dim, n_steps_per_bar, n_pitches)
            )
        # self.bar_generator = [None] * self.n_tracks
        # for track_idx in range(self.n_tracks):
        #     self.bar_generator[track_idx] = BarGenerator(z_dim, n_steps_per_bar, n_pitches)
            
    def forward(self, chords, style, melody, groove):
        """_summary_

        Args:
            chords (_type_): shape: (batch_size, z_dim)
            style (_type_): shape: (batch_size, z_dim)
            melody (_type_): shape: (batch_size, n_tracks, z_dim)
            groove (_type_): shape: (batch_size, n_tracks, z_dim)

        Returns:
            _type_: _description_
        """
        # 화음
        # shape: (batch_size, z_dim) -> (batch_size, z_dim, n_bars)
        chords_over_time = self.chord_temp_network(chords)
        
        # 멜로디
        # shape: (batch_size, n_tracks, z_dim) -> [n_tracks]:(batch_size, z_dim, n_bars)
        melody_over_time = []
        for track_idx in range(self.n_tracks):
            melody_track = melody[:, track_idx, :]
            melody_over_time.append(self.melody_temp_networks["melodygen_" + str(track_idx)](melody_track))
            
        # 트랙과 마디마다 출력을 생성합니다
        # c: (batch_size, z_dim)
        # s: (batch_size, z_dim)
        # m: (batch_size, z_dim)
        # g: (batch_size, z_dim)
        # z_input: (batch_size, 4 * z_dim)
        # track_output: [n_tracks]: (batch_size, 1, 1, n_steps_per_bar, n_pitches)
        # bars_output: [n_bars]: (batch_size, n_tracks, 1, n_steps_per_bar, n_pitches)
        # generator_output: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)
        bars_output = []
        for bar_idx in range(self.n_bars):
            track_output = []
            
            c = chords_over_time[:, :, bar_idx]
            s = style
            
            for track_idx in range(self.n_tracks):
                m = melody_over_time[track_idx][:, :, bar_idx]
                g = groove[:, track_idx, :]
                
                z_input = torch.cat((c, s, m, g), dim=1)
                
                track_output.append(self.bar_generators["bargen_" + str(track_idx)](z_input))
                
            bars_output.append(torch.cat(track_output, dim=1))
            
        generator_output = torch.cat(bars_output, dim=2)
        
        return generator_output
    
    
    def notes_to_midi(self, filepath, output):
        for score_num in range(len(output)):
            # output: [batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches]
            max_pitches = np.argmax(output, axis=-1)
            midi_note_score = max_pitches[score_num].reshape(self.n_tracks, self.n_bars * self.n_steps_per_bar)
            parts = stream.Score()
            parts.append(tempo.MetronomeMark(number=66))
            
            for i in range(self.n_tracks):
                last_x = int(midi_note_score[i, :][0])
                s = stream.Part()
                dur = 0
                
                for idx, x in enumerate(midi_note_score[i, :]):
                    x = int(x)
                    
                    if(x != last_x or idx % 4 == 0) and idx > 0:
                        n = note.Note(last_x)
                        n.duration = duration.Duration(dur)
                        s.append(n)
                        dur = 0
                        
                    last_x = x
                    dur = dur + 0.25
                    
                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)
                
                parts.append(s)
                
            parts.write('midi', fp=filepath)

    def draw_score(self, data, score_num):
        fig, axes = plt.subplots(ncols=self.n_bars, nrows=self.n_tracks, figsize=(12,8), sharey=True, sharex=True)
        fig.subplots_adjust(0, 0, 0.2, 1.5, 0, 0)

        for track in range(self.n_tracks):
            for bar in range(self.n_bars):
                if self.n_bars > 1:
                    axes[track, bar].imshow(data[score_num, track, bar, :, :].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)
                else:
                    axes[track].imshow(data[score_num, track, bar, :, :].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)
    

class Critic(nn.Module):
    def __init__(self, in_channels, n_bars):
        super(Critic, self).__init__()
        
        self.in_channels = in_channels
        self.n_bars = n_bars
        self.n_layers = 8
        self.channels = [in_channels, 128, 128, 128, 128, 128, 128, 256, 512]
        self.kernel_sizes = [(2, 1, 1), (self.n_bars - 1, 1, 1), (1, 1, 12), (1, 1, 7), (1, 2, 1), (1, 2, 1), (1, 4, 1), (1, 3, 1)]
        self.strides = [(1, 1, 1), (1, 1, 1), (1, 1, 12), (1, 1, 7), (1, 2, 1), (1, 2, 1), (1, 2, 1), (1, 2, 1)]
        self.paddings = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 0)]
        self.activation_functions = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu']
        
        self.layers = self._make_layers()
        
        self.linear1 = nn.Linear(self.channels[-1], 1024)
        self.leaky_relu = nn.LeakyReLU(0.3, inplace=True)
        self.linear2 = nn.Linear(1024, 1)
        
        # nn.init.normal_(self.linear1.weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.linear2.weight, mean=0.0, std=0.02)
        
    def _make_layers(self):
        layers = nn.Sequential()
        for i in range(self.n_layers):
            layers.add_module(f'conv3d_block_{i + 1}', Conv3dBlock(self.channels[i],
                                                                   self.channels[i + 1],
                                                                   self.kernel_sizes[i],
                                                                   self.strides[i],
                                                                   self.paddings[i],
                                                                   self.activation_functions[i]))
        return layers
        
    def forward(self, x):
        # x shape: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)
        # out shape: (batch_size, 512, 1, 1, 1)
        out = self.layers(x)
        # out shape: (batch_size, 512)
        out = out.view(out.size(0), -1)
        # out shape: (batch_size, 1024)
        out = self.linear1(out)
        out = self.leaky_relu(out)
        # out shape: (batch_size, 1)
        out = self.linear2(out)
        
        return out