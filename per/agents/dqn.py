import torch as tc


class QNetwork(tc.nn.Module):
    def __init__(self, architecture, head):
        super().__init__()
        self._architecture = architecture
        self._head = head

    def forward(self, x):
        features = self._architecture(x)
        qpred = self._head(features)
        return qpred

    def sample(self, x, epsilon):
        probs = tc.tile(
            tc.FloatTensor([1-epsilon, epsilon]).unsqueeze(0),
            [x.shape[0], 1]
        )
        do_rand = tc.distributions.Categorical(probs=probs).sample()
        greedy_action = tc.argmax(self.forward(x), dim=-1)
        random_action = tc.randint(
            low=0, high=self._num_actions, size=x.shape[0])

        action = (1-do_rand) * greedy_action + do_rand * random_action
        return action
