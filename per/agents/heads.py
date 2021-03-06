import torch as tc


class LinearActionValueHead(tc.nn.Module):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self._num_features = num_features
        self._num_actions = num_actions
        self._linear = tc.nn.Linear(
            in_features=self._num_features,
            out_features=self._num_actions,
            bias=True)

    @property
    def num_actions(self):
        return self._num_actions

    def forward(self, x):
        qpred = self._linear(x)
        return qpred


class DuelingActionValueHead(tc.nn.Module):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self._num_features = num_features
        self._num_actions = num_actions

        self._value_head = tc.nn.Sequential(
            tc.nn.Linear(self._num_features, self._num_features, bias=True),
            tc.nn.ReLU(),
            tc.nn.Linear(self._num_features, 1, bias=True)
        )

        self._advantage_head = tc.nn.Sequential(
            tc.nn.Linear(self._num_features, self._num_features, bias=True),
            tc.nn.ReLU(),
            tc.nn.Linear(self._num_features, self._num_actions, bias=False)
        )

    @property
    def num_actions(self):
        return self._num_actions

    def forward(self, x):
        vpred = self._value_head(x)
        apred = self._advantage_head(x)
        apred -= apred.mean(dim=-1).unsqueeze(-1)
        qpred = vpred + apred
        return qpred
