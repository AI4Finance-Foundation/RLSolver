
class BinaryKnapsackEnv(KnapsackEnv):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)
        assign_env_config(self, kwargs)

        obs_space = spaces.Box(
            0, self.max_weight, shape=(3, self.N + 1), dtype=np.int32)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(len(self.item_limits),)),
                "avail_actions": spaces.Box(0, 1, shape=(len(self.item_limits),)),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space

        self.reset()

    def _STEP(self, item):
        # Check item limit
        if self.item_limits[item] > 0:
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

    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
        state_items = np.vstack([
            self.item_weights,
            self.item_values,
            self.item_limits
        ])
        state = np.hstack([
            state_items, 
            np.array([[self.max_weight],
                      [self.current_weight], 
                      [0] # Serves as place holder
                ])
        ])
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight,
                0, 1)
            mask = np.where(self.item_limits > 0, mask, 0)
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N),
                "state": state
            }
        else:
            self.state = state.copy()
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self.item_limits = np.ones(self.N)
        self._update_state()
        return self.state
