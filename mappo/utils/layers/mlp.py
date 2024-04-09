import torch.nn as nn
from .util import init, get_clones

from .noisy import NoisyLinear

"""MLP modules."""


class MLPLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, noise_type=None):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        self.noise_type = noise_type

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if noise_type == "fixed":
            self.fc1 = nn.Sequential(NoisyLinear(input_dim, hidden_size), active_func, nn.LayerNorm(hidden_size))
            self.fc_h = nn.Sequential(NoisyLinear(hidden_size, hidden_size), active_func, nn.LayerNorm(hidden_size))
            self.fc2 = get_clones(self.fc_h, self._layer_N)
        elif noise_type == "adaptive":
            self.fc1 = nn.Sequential(
                NoisyLinear(input_dim, hidden_size, std_init=0.1), active_func, nn.LayerNorm(hidden_size)
            )
            self.fc_h = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size, std_init=0.1), active_func, nn.LayerNorm(hidden_size)
            )
            self.fc2 = get_clones(self.fc_h, self._layer_N)
        else:
            self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            self.fc_h = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)
            )
            self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

    def resample(self):
        self.fc1[0].resample()
        for i in range(self._layer_N):
            self.fc2[i][0].resample()


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, noise_type=None, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU, noise_type)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x

    def resample(self):
        self.mlp.resample()
