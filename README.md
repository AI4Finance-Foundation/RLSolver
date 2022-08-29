# RLSolver: 

Massively Parallel Gym-Environments for Combinatorial and Nonconvex Optimizations
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