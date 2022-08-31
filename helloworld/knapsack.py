import numpy as np
class BinaryKnapsack():
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.N = 5
        self.max_weight = 15
        self.current_weight = 0
        self._max_reward = 10000
        self.item_numbers = np.arange(self.N)
        self.item_weights = np.array([1, 12, 2, 1, 4]) # np.random.randint(1, 5, size=self.N)
        self.item_values = np.array([2, 4, 2, 1, 10]) # np.random.randint(0, 100, size=self.N)
        self.randomize_params_on_reset = False
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)
        self.state_dim = 3 * (self.N + 1)
        self.action_dim = self.N

    def step(self, item):
        # Check item limit
        if self.item_indicator[item] > 0:
            # Check that item will fit
            if self.item_weights[item] + self.current_weight <= self.max_weight:
                self.current_weight += self.item_weights[item]
                reward = self.item_values[item]
                if self.current_weight == self.max_weight:
                    done = True
                else:
                    done = False
                self._update_state(item)
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            # End if item is unavailable
            reward = 0
            done = True          
        return self.state, reward, done, {}

    def get_state(self, item=None):
        if item is not None:
            self.item_indicator[item] -= 1
        state_items = np.vstack([self.item_weights, self.item_values, self.item_indicator])
        state = np.hstack([state_items, np.array([[self.max_weight], [self.current_weight], [0]])])
        self.state = state
    
    def reset(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self.item_indicator = np.ones(self.N)
        self._update_state()
        return self.state
