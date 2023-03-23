hidden_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
cell_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]

class OptimizerMeta(nn.Module):
    def __init__(self, hid_dim=20):
        super().__init__()
        self.hid_dim = hid_dim
        self.recurs = nn.LSTMCell(1, hid_dim)
        self.recurs2 = nn.LSTMCell(hid_dim, hid_dim)
        self.output = nn.Linear(hid_dim, 1)

    def forward(self, inp, hidden, cell):
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

opt_net = OptimizerMeta(hid_dim=20)

updates, new_hidden, new_cell = opt_net(
    gradients,
    [h[offset:offset + cur_sz] for h in hidden_states],
    [c[offset:offset + cur_sz] for c in cell_states]
)
