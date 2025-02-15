import os
import torch
import numpy as np
import random
from utils import events_to_event_frame, quad_bayer_to_rgb_d2

class EventData():
    def __init__(self, data_path, t_start, t_end, H=260, W=346, color_event=True, event_thresh=1, device='cuda'):
        self.data_path = data_path
        self.t_start = t_start
        self.t_end = t_end
        self.H, self.W = H, W
        self.color_event = color_event
        self.event_thresh = event_thresh
        self.device = device

        self.events = self.load_events()
        
    def load_events(self):
        events = np.load(self.data_path)
        events[: ,0] = events[:, 0] - events[0, 0]
        events[events[:, 3] == 0, 3] = -1
        events = events[(events[:, 0] > self.t_start) & (events[:, 0] < self.t_end)]
        events[: ,0] = (events[: ,0] - self.t_start) / (self.t_end - self.t_start)# Normalize event timestampes to [0, 1]

        if self.H > self.W:
            self.H, self.W = self.W, self.H
            events = events[:, [0, 2, 1, 3]]
            
        if events.shape[0] == 0:
            raise ValueError(f'No events in [{self.t_start}, {self.t_end}]!')
        #print(f'Loaded {events.shape[0]} events in [{self.t_start}, {self.t_end}] ...')
        #print(f'First event: {events[0]}')
        #print(f'Last event: {events[-1]}')
        return events

    def stack_event_frames(self, num_frames):
        print(f'Stacking {num_frames} event frames from {self.events.shape[0]} events ...')
        event_chunks = np.array_split(self.events, num_frames)
        event_frames, event_timestamps,event_frames_left,event_frames_right,event_timestamps_middle = [], [],[], [],[]
        for i, event_chunk in enumerate(event_chunks):
            event_frame = events_to_event_frame(event_chunk, self.H, self.W).squeeze(-1)
            if self.color_event:
                event_frame = quad_bayer_to_rgb_d2(event_frame)
            event_frame *= self.event_thresh
            event_frames.append(event_frame)
            event_timestamps.append(event_chunk[0, 0])# + event_chunk[-1, 0]) / 2)
            #adding the middle timestamp and the right/left frame for stage 2 loss calculation 
            t_mark = random.randint(1, len(event_chunk))
            event_frame_left = events_to_event_frame(event_chunk[0:t_mark, :], self.H, self.W).squeeze(-1)
            event_frame_left *= self.event_thresh
            event_frame_right = events_to_event_frame(event_chunk[t_mark:-1,:], self.H, self.W).squeeze(-1)
            event_frame_right *= self.event_thresh
            timestamps_middle = event_chunk[t_mark, 0]
            event_frames_left.append(event_frame_left)
            event_frames_right.append(event_frame_right)
            event_timestamps_middle.append(timestamps_middle)

        frames_left = np.stack(event_frames_left, axis=0).reshape(num_frames, self.H, self.W, -1)
        self.event_frames_left = torch.as_tensor(frames_left).float().to(self.device)
        frames_right = np.stack(event_frames, axis=0).reshape(num_frames, self.H, self.W, -1)
        self.event_frames_right = torch.as_tensor(frames_right).float().to(self.device)
        event_frames = np.stack(event_frames, axis=0).reshape(num_frames, self.H, self.W, -1)
        self.event_frames = torch.as_tensor(event_frames).float().to(self.device)

        timestamps = np.stack(event_timestamps, axis=0).reshape(num_frames, 1)
        self.timestamps = torch.as_tensor(timestamps).float().to(self.device)

        timestamps_middle = np.stack(event_timestamps_middle, axis=0).reshape(num_frames, 1)
        self.event_timestamps_middle = torch.as_tensor(timestamps_middle).float().to(self.device)
'''    
    def stack_event_frames(self, num_frames):
        print(f'Stacking {num_frames} event frames from {self.events.shape[0]} events ...')
        event_chunks = np.array_split(self.events, num_frames)
        event_frames, event_timestamps = [], []
        for i, event_chunk in enumerate(event_chunks):
            event_frame = events_to_event_frame(event_chunk, self.H, self.W).squeeze(-1)
            if self.color_event:
                event_frame = quad_bayer_to_rgb_d2(event_frame)
            event_frame *= self.event_thresh
            event_frames.append(event_frame)
            event_timestamps.append(event_chunk[0, 0])# + event_chunk[-1, 0]) / 2)
        
        event_frames = np.stack(event_frames, axis=0).reshape(num_frames, self.H, self.W, -1)
        self.event_frames = torch.as_tensor(event_frames).float().to(self.device)
        timestamps = np.stack(event_timestamps, axis=0).reshape(num_frames, 1)
        self.timestamps = torch.as_tensor(timestamps).float().to(self.device)
'''        

