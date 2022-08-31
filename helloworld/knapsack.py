class KnapsackEnv(gym.Env):
    # Internal list of placed items for better rendering
    _collected_items = []
    
    def __init__(self, *args, **kwargs):
        self.N = 5
        self.max_weight = 15
        self.current_weight = 0
        self._max_reward = 10000
        self.mask = False
        self.seed = 0
        self.item_numbers = np.arange(self.N)
        self.item_weights = np.array([1, 12, 2, 1, 4]) # np.random.randint(1, 5, size=self.N)
        self.item_values = np.array([2, 4, 2, 1, 10])  # np.random.randint(0, 100, size=self.N)
        self.over_packed_penalty = 0
        self.randomize_params_on_reset = False
        self._collected_items.clear()
        assign_env_config(self, kwargs)
        self.set_seed()

        obs_space = spaces.Box(0, self.max_weight, shape=(2*self.N + 1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.N)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.N,), dtype=np.int32),
                "avail_actions": spaces.Box(0, 1, shape=(self.N,), dtype=np.int16),
                "state": obs_space
                })
        else:
            self.observation_space = spaces.Box(
                0, self.max_weight, shape=(2, self.N + 1), dtype=np.int32)
        
        self.reset()
        
    def _STEP(self, item):
        # Check that item will fit
        if self.item_weights[item] + self.current_weight <= self.max_weight:
            self.current_weight += self.item_weights[item]
            reward = self.item_values[item]
            self._collected_items.append(item)
            if self.current_weight == self.max_weight:
                done = True
            else:
                done = False
        else:
            # End trial if over weight
            reward = self.over_packed_penalty
            done = True
            
        self._update_state()
        return self.state, reward, done, {}
    
    def _get_obs(self):
        return self.state
    
    def _update_state(self):
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight, 0, 1)
            state = np.hstack([
                self.item_weights,
                self.item_values,
                np.array([self.current_weight])
                ])
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N, dtype=np.int16),
                "state": state
                }
        else:
            state = np.vstack([
                self.item_weights,
                self.item_values])
            self.state = np.hstack([
                state,
                np.array([
                    [self.max_weight],
                     [self.current_weight]])
                ])        
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self._collected_items.clear()
        self._update_state()
        return self.state
    
    def sample_action(self):
        return np.random.choice(self.item_numbers)

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self._RESET()

    def step(self, action):
        return self._STEP(action)
        
    def render(self):
        total_value = 0
        total_weight = 0
        for i in range(self.N) :
            if i in self._collected_items :
                total_value += self.item_values[i]
                total_weight += self.item_weights[i]
        print(self._collected_items, total_value, total_weight)
        
        return True
