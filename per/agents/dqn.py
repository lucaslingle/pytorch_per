import torch as tc


class QNetwork(tc.nn.Module):
    """
    Wrapper class integrating a vision network and an action-value head.
    """
    def __init__(self, architecture, head):
        super().__init__()
        self._architecture = architecture
        self._head = head

    def forward(self, x):
        features = self._architecture(x)
        qpred = self._head(features)
        return qpred

    def sample(self, x, epsilon):
        batch_size = x.shape[0]
        probs = tc.FloatTensor([1-epsilon, epsilon])
        dist = tc.distributions.Categorical(probs=probs)
        do_rand = dist.sample((batch_size,))

        greedy_action = tc.argmax(self.forward(x).detach(), dim=-1)
        random_action = tc.randint(
            low=0, high=self._num_actions, size=batch_size)

        action = (1-do_rand) * greedy_action + do_rand * random_action
        return action