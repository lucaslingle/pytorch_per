import torch as tc


class QNetwork(tc.nn.Module):
    """
    Wrapper class integrating preprocessing, vision net, and action-value head.
    """
    def __init__(self, architecture, head):
        super().__init__()
        self._architecture = architecture
        self._head = head

    def forward(self, x):
        x = x / 255.
        x = x.permute(0, 3, 1, 2)
        features = self._architecture(x)
        qpred = self._head(features)
        return qpred

    def sample(self, x, epsilon, device):
        batch_size = x.shape[0]
        probs = tc.FloatTensor([1-epsilon, epsilon])
        dist = tc.distributions.Categorical(probs=probs)
        do_rand = dist.sample((batch_size,)).to(device)

        qpred = self.forward(x)
        greedy_action = tc.argmax(qpred, dim=-1)
        random_action = tc.randint(
            low=0, high=self._head.num_actions, size=(batch_size,)).to(device)

        action = (1-do_rand) * greedy_action + do_rand * random_action
        return action
