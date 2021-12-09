import torch as tc
import numpy as np


def normc_initializer(weight_tensor, gain=1.0):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L97

    Note that in tensorflow the weight tensor in a linear layer is stored with
    the input dim first and the output dim second. In pytorch, the output dim is first,

    We currently only support normc init for linear layers.
    Performance not guaranteed with other layer weight types.
    """
    with tc.no_grad():
        out = np.random.normal(loc=0.0, scale=1.0, size=weight_tensor.size())
        out = gain * out / np.sqrt(np.sum(np.square(out), axis=1, keepdims=True))
        weight_tensor.copy_(tc.tensor(out))


class NatureCNN(tc.nn.Module):
    """
    Implements the convolutional torso of the agent from Mnih et al., 2015
    - 'Human Level Control through Deep Reinforcement Learning'.
    """
    def __init__(self):
        super().__init__()
        self._feature_dim = 512
        self._network = tc.nn.Sequential(
            tc.nn.Conv2d(4, 32, kernel_size=(8,8), stride=(4,4)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1)),
            tc.nn.ReLU(),
            tc.nn.Flatten(),
            tc.nn.Linear(3136, self._feature_dim, bias=True),
            tc.nn.ReLU()
        )

    @property
    def output_dim(self):
        return self._feature_dim

    def forward(self, x):
        features = self._network(x)
        return features
