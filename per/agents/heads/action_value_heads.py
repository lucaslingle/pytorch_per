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
        self._init_weights()

    def _init_weights(self):
        tc.nn.init.xavier_uniform_(self._linear.weight)

    def forward(self, x):
        qpred = self._linear(x)
        return qpred


class DuelingActionValueHead(tc.nn.Module):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self._num_features = num_features
        self._num_actions = num_actions

        self._value_head = tc.nn.Sequential(
            tc.nn.Linear(
                in_features=self._num_features,
                out_features=self._num_actions,
                bias=True),
            tc.nn.ReLU(),
            tc.nn.Linear(
                in_features=self._num_features,
                out_features=1,
                bias=True)
        )

        self._advantage_head = tc.nn.Sequential(
            tc.nn.Linear(
                in_features=self._num_features,
                out_features=self._num_actions,
                bias=True),
            tc.nn.ReLU(),
            tc.nn.Linear(
                in_features=self._num_features,
                out_features=self._num_actions,
                bias=False)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self._value_head:
            if isinstance(m, tc.nn.Linear):
                tc.nn.init.xavier_normal_(m.weight)
        for m in self._advantage_head:
            if isinstance(m, tc.nn.Linear):
                tc.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        vpred = self._value_head(x).unsqueeze(-1)
        apred = self._advantage_head(x)
        apred -= apred.mean(dim=-1).unsqueeze(-1)
        qpred = vpred + apred
        return qpred
