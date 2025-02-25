import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """This is the backbone that supports DQN for gym simple games
    (The ones where the observation space is not an image)"""

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


class Nature_CNN(nn.Module):
    """This is the backbone that supports DQN for games where the \
        observation space is composed of images"""

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        self.n_features = 512

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(1024, self.n_features)
        self._h5 = nn.Linear(self.n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h5.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action=None):
        x = torch.permute(state, (0, 3, 1, 2))
        h = F.relu(self._h1(x.float() / 255.0))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.reshape(-1, 1024)))
        q = self._h5(h)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


class ImpalaCNN(nn.Module):
    """This is the IMPALA CNN architecture with *4 channels. It can be used as backbone for
    Neural Network function approximators such as DQN and Rainbow"""

    def __init__(self, input_shape, output_shape, depths=[16, 32, 32], **kwargs):
        super(ImpalaCNN, self).__init__()

        self.depths = depths
        self.layer_num = 0
        self.input_channels = input_shape[-1]
        n_output = output_shape[0]

        def get_layer_num_str():
            layer_str = str(self.layer_num)
            self.layer_num += 1
            return layer_str

        def conv_layer(in_channels, out_channels):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        def residual_block(depth):
            layers = []
            layers.append(nn.ReLU())
            layers.append(conv_layer(depth, depth))
            layers.append(nn.ReLU())
            layers.append(conv_layer(depth, depth))
            return nn.Sequential(*layers)

        def conv_sequence(in_channels, out_channels):
            layers = []
            layers.append(conv_layer(in_channels, out_channels))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(residual_block(out_channels))
            layers.append(residual_block(out_channels))
            return nn.Sequential(*layers)

        self.conv_layers = nn.ModuleList()
        for depth in depths:
            self.conv_layers.append(conv_sequence(self.input_channels, depth * 4))
            self.input_channels = depth * 4

        self.fc = nn.Linear(2048 * 4, n_output)

    def forward(self, x, action=None):
        # This is the permutation for procgen
        x = torch.permute(x, (0, 3, 1, 2))

        x = x.float() / 255.0

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(x.size(0), -1)
        q = F.relu(self.fc(x))

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted
