import torch as th


class PortfolioOptEnv():
    def __init__(self, cost_transaction, asset_price_mean, asset_price_var, num_assets=3, initial_cash=100, num_env=1024, episode_length=6, device=th.device("cpu")):
        self.num_env = num_env
        self.num_assets = num_assets # Number of assets 
        self.initial_cash = initial_cash # Starting amount of capital 
        self.episode_length = episode_length
        #Transaction costs proportional to amount bought 
        self.cost_transaction = cost_transaction
        # Prices of assets have a mean value in every period and vary according to a Gaussian distribution 
        self.price_asset_mean = asset_price_mean
        self.price_asset_var = asset_price_var
        self.state_dim = 1 + 2 * self.num_assets  # Cash on hand, asset prices, num of shares
        self.action_dim = self.num_assets
        
    def reset(self, ):
        self.num_steps = 0
        self.price_asset = th.randn(self.episode_length, self.price_asset_mean, self.price_asset_var)
        self.holdings = th.zeros(self.num_env, self.num_assets)
        self.cash = self.initial_cash
        self.state = th.cat((self.cash, self.asset_prices[0], self.holdings), dim=1)
        return self.state
    
    def step(self, action):
        action += self.holdings[(self.holdings + action) < 0]
        self.holdings += action
        self.cash -= th.mul(self.price_asset[self.num_steps], action[action > 0]).sum(dim=1) * (1 + self.cost_transaction)
        self.cash += th.mul(self.price_asset[self.num_steps], action[action < 0]).sum(dim=1) * (1 - self.cost_transaction)
        self.reward = self.cash + th.matmul(self.holdings, self.price_asset).sum(dim=1)
        self.done = self.cash[self.cash < 0]
        self.num_steps += 1
        self.state = th.cat((self.cash, self.asset_prices[self.num_steps], self.holdings), dim=1)
        return self.state, self.reward, self.done
            
        
        