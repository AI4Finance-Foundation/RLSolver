# RLSolver: 

Massively Parallel Gym-Environments for Combinatorial and Nonconvex Optimizations

# FinRL: Financial Reinforcement Learning [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]

[1.1]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_twitter_22x22.png
[1.2]: http://www.tensorlet.org/wp-content/uploads/2021/01/facebook-button_22x22.png
[1.3]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_google_22.xx_.png
[1.4]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_linkedin_22x22.png

[1]: https://twitter.com/intent/tweet?text=FinRL-Financial-Deep-Reinforcement-Learning%20&url=https://github.com/AI4Finance-Foundation/FinRL&hashtags=DRL&hashtags=AI
[2]: https://www.facebook.com/sharer.php?u=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL
[3]: https://plus.google.com/share?url=https://github.com/AI4Finance-Foundation/FinRL
[4]: https://www.linkedin.com/sharing/share-offsite/?url=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="55%"/>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)



**FinRL** ([website](https://finrl.readthedocs.io/en/latest/index.html)) is **the first open-source framework** to show the great potential of financial reinforcement learning.

FinRL has evolving into an ecosystem, including hundreds of financial markets, state-of-the-art algorithms, financial applications (portfolio allocation, cryptocurrency trading, high-frequency trading), live trading, cloud deployment, etc.

| Roadmap  | Level | Target Users | Example | Desription |
|----|----|----|----|----|
| 0.0 (Preparation) | preparation | practitioners | [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Metaverse)| a playground |
| 1.0 (Proof-of-Concept)| entry-level | beginners | [this repo](https://github.com/AI4Finance-Foundation/FinRL) | demonstration, education |
| 2.0 (Professional) | intermediate-level  | full-stack developers, professionals | [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | financially optimized DRL algorithms |
| 3.0 (Production) | advance-level | investment banks, hedge funds | [Podracer](https://github.com/AI4Finance-Foundation/FinRL_Podracer) | cloud-native solutions |


## Outline

- [File Structure](#File Structure)

## File Structure

```
RLSolver
├── finrl (main folder)
│   ├── applications
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	└── stock_trading
│   ├── agents
│   	├── elegantrl
│   	├── rllib
│   	└── stablebaseline3
│   ├── meta
│   	├── data_processors
│   	├── env_cryptocurrency_trading
│   	├── env_portfolio_allocation
│   	├── env_stock_trading
│   	├── preprocessor
│   	├── data_processor.py
│       ├── meta_config_tickers.py
│   	└── meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
│
├── tutorials (educational notebook files)
├── tests (unit tests to verify codes on env & data)
│   ├── environments
│   	└── test_env_cashpenalty.py
│   └── downloaders
│   	├── test_yahoodownload.py
│   	└── test_alpaca_downloader.py
├── setup.py
├── requirements.txt
└── README.md
```