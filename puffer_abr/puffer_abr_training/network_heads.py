import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class SmallCNN(BaseFeaturesExtractor):
    """
        Conv-1D extractor for ABR
    """

    def __init__(self, observation_space: gym.spaces.Box, 
            features_dim: int = 256, history_len: int = 10, 
            network_features: int = 6):
        super(SmallCNN, self).__init__(observation_space, features_dim=features_dim)
        self.network_features = network_features
        self.history_len = history_len
        self.network_cnn = nn.Sequential(
            nn.Conv1d(self.history_len, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),   
            nn.Flatten())
        self.video_cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten())
        self.quality_cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        network_features = self.network_cnn(
            observations[..., 0:self.history_len, 0:self.network_features])
        video_features = self.video_cnn(
            observations[..., 0:5, self.network_features:self.network_features+10])
        quality_features = self.quality_cnn(
            observations[..., 0:5, self.network_features+10:self.network_features+20])
        features = th.cat((network_features, video_features, quality_features), dim=1)
        return features



class PensieveExtractorOriginal(BaseFeaturesExtractor):
    """
        Conv-1D extractor for ABR
    """

    def __init__(self, observation_space: gym.spaces.Box, 
            features_dim: int =3072,
            future_horizon: int=0, history_len: int=10):
        super(PensieveExtractorOriginal, self).__init__(observation_space, features_dim)
        self.future_horizon = future_horizon + 1
        self.history_len = history_len
        self.quality_net = nn.Sequential(nn.Linear(1, 128), nn.ReLU())
        self.buffer_net = nn.Sequential(nn.Linear(1, 128), nn.ReLU())
        self.throughput_net = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4, stride=1), nn.ReLU(),nn.Flatten())
        self.delay_net = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4, stride=1), nn.ReLU(), nn.Flatten())
        self.video_size_net = nn.Sequential(
            nn.Conv1d(self.future_horizon, 128, kernel_size=4, stride=1),
            nn.ReLU(), nn.Flatten())
        self.index_net = nn.Sequential(nn.Linear(1, 128), nn.ReLU())
        self.throughput_indices = (2, 2+history_len)
        self.delay_indices = (self.throughput_indices[1], 
            self.throughput_indices[1] + history_len)
        self.video_indices = (self.delay_indices[1], 
            self.delay_indices[1] + 10*self.future_horizon)
        self.n_index = self.video_indices[1]
     


    def forward(self, observations: th.Tensor) -> th.Tensor:
        quality_out = self.quality_net(observations[...,0, None])
        buffer_out = self.buffer_net(observations[...,1, None])
        throughput_in = observations[...,
            self.throughput_indices[0]: self.throughput_indices[1]].view(-1, 1, self.history_len)
        throughput_out = self.throughput_net(throughput_in)
        delay_in = observations[...,
            self.delay_indices[0]: self.delay_indices[1]].view(-1, 1, self.history_len)
        delay_out = self.delay_net(delay_in)
        video_in = observations[...,
            self.video_indices[0]: self.video_indices[1]].view(-1, self.future_horizon, 10)
        video_out = self.video_size_net(video_in)
        index_out = self.index_net(observations[...,
            self.n_index, None])
        output = th.cat((quality_out, buffer_out,  
            throughput_out, delay_out, video_out, index_out), dim=1)
        
        return output