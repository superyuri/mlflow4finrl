# region 初始化程序
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''

import mlflow
import private_functions
import tools

# 初始化参数
args = private_functions.init_args()
# 拦截文本输出
tools.init_stdout()

# 如果是ipython运行，创建MLflow上下文
is_mlflow_project_mode = tools.is_mlflow_project_mode()
if(not is_mlflow_project_mode):
  print("WORKING MODE: IPYTHON")
  mlflow.start_run()
  # 输出参数
  for key, value in vars(args).items():
      mlflow.log_param(key, value)
else:
  print("WORKING MODE: MLFLOW PROJECT")
   

import mlflow
import sys


'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion

#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL/blob/master/Stock_NeurIPS2018.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading
# 
# * **Pytorch Version** 
# 
# 

# # Content

# * [1. Task Description](#0)
# * [2. Install Python packages](#1)
#     * [2.1. Install Packages](#1.1)    
#     * [2.2. A List of Python Packages](#1.2)
#     * [2.3. Import Packages](#1.3)
#     * [2.4. Create Folders](#1.4)
# * [3. Download and Preprocess Data](#2)
# * [4. Preprocess Data](#3)        
#     * [4.1. Technical Indicators](#3.1)
#     * [4.2. Perform Feature Engineering](#3.2)
# * [5. Build Market Environment in OpenAI Gym-style](#4)  
#     * [5.1. Data Split](#4.1)  
#     * [5.3. Environment for Training](#4.2)    
# * [6. Train DRL Agents](#5)
# * [7. Backtesting Performance](#6)  
#     * [7.1. BackTestStats](#6.1)
#     * [7.2. BackTestPlot](#6.2)   
#   

# <a id='0'></a>
# # Part 1. Task Discription

# We train a DRL agent for stock trading. This task is modeled as a Markov Decision Process (MDP), and the objective function is maximizing (expected) cumulative return.
# 
# We specify the state-action-reward as follows:
# 
# * **State s**: The state space represents an agent's perception of the market environment. Just like a human trader analyzing various information, here our agent passively observes many features and learns by interacting with the market environment (usually by replaying historical data).
# 
# * **Action a**: The action space includes allowed actions that an agent can take at each state. For example, a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying. When an action operates multiple shares, a ∈{−k, ..., −1, 0, 1, ..., k}, e.g.. "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
# 
# * **Reward function r(s, a, s′)**: Reward is an incentive for an agent to learn a better policy. For example, it can be the change of the portfolio value when taking a at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio values at state s′ and s, respectively
# 
# 
# **Market environment**: 30 consituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at the starting date of the testing period.
# 
# 
# The data for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
# 

# <a id='1'></a>
# # Part 2. Install Python Packages

# <a id='1.1'></a>
# ## 2.1. Install packages
# 

# In[56]:


# region 在MLFlow中初始化，注释掉此命令
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''
# get_ipython().system('pip install wrds')
# get_ipython().system('pip install swig')
# get_ipython().system('pip install git+https://github.com/AI4Finance-Foundation/FinRL.git')

'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion

# 
# <a id='1.2'></a>
# ## 2.2. A list of Python packages 
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio

# <a id='1.3'></a>
# ## 2.3. Import Packages

# In[57]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

import sys
sys.path.append("../FinRL")

import itertools


# <a id='1.4'></a>
# ## 2.4. Create Folders

# In[58]:


from finrl import config
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])



# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance provides stock data, financial news, financial reports, etc. Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** in FinRL-Meta to fetch data via Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).

# 
# 
# -----
# class YahooDownloader:
#     Retrieving daily stock data from
#     Yahoo Finance API
# 
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
# 
#     Methods
#     -------
#     fetch_data()
# 

# In[59]:


# from config.py, TRAIN_START_DATE is a string
TRAIN_START_DATE
# from config.py, TRAIN_END_DATE is a string
TRAIN_END_DATE


# In[60]:

# region 修改为从入参导入
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''

# TRAIN_START_DATE = '2009-01-01'
# TRAIN_END_DATE = '2020-07-01'
# TRADE_START_DATE = '2020-07-01'
# TRADE_END_DATE = '2021-10-31'

TRAIN_START_DATE = args.train_start_date
TRAIN_END_DATE = args.train_end_date
TRADE_START_DATE = args.trade_start_date
TRADE_END_DATE = args.trade_end_date

'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion

# In[61]:


df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()


# In[62]:


print(config_tickers.DOW_30_TICKER)


# In[63]:


df.shape


# In[64]:


df.sort_values(['date','tic'],ignore_index=True).head()


# # Part 4: Preprocess Data
# We need to check for missing data and do feature engineering to convert the data point into a state.
# * **Adding technical indicators**. In practical trading, various information needs to be taken into account, such as historical prices, current holding shares, technical indicators, etc. Here, we demonstrate two trend-following technical indicators: MACD and RSI.
# * **Adding turbulence index**. Risk-aversion reflects whether an investor prefers to protect the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the turbulence index that measures extreme fluctuation of asset price.

# In[65]:


fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)


# In[66]:


list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)


# In[67]:


processed_full.sort_values(['date','tic'],ignore_index=True).head(10)


# <a id='4'></a>
# # Part 5. Build A Market Environment in OpenAI Gym-style
# The training process involves observing stock price change, taking an action and reward's calculation. By interacting with the market environment, the agent will eventually derive a trading strategy that may maximize (expected) rewards.
# 
# Our market environment, based on OpenAI Gym, simulates stock markets with historical market data.

# ## Data Split
# We split the data into training set and testing set as follows:
# 
# Training data period: 2009-01-01 to 2020-07-01
# 
# Trading data period: 2020-07-01 to 2021-10-31
# 

# In[68]:


train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
print(len(train))
print(len(trade))


# In[69]:


train.tail()


# In[70]:


trade.head()


# In[71]:


INDICATORS


# In[72]:


stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[74]:


buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs)


# ## Environment for Training
# 
# 

# In[75]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# <a id='5'></a>
# # Part 6: Train DRL Agents
# * The DRL algorithms are from **Stable Baselines 3**. Users are also encouraged to try **ElegantRL** and **Ray RLlib**.
# * FinRL includes fine-tuned standard DRL algorithms, such as DQN, DDPG, Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.

# In[76]:


agent = DRLAgent(env = env_train)

if_using_a2c = False
if_using_ddpg = False
if_using_ppo = False
if_using_td3 = False
if_using_sac = True


# ### Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC)
# 

# ### Agent 1: A2C
# 

# In[77]:


agent = DRLAgent(env = env_train)
model_a2c = agent.get_model("a2c")

if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_a2c.set_logger(new_logger_a2c)


# In[78]:

# region 修改为从入参导入
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''
# trained_a2c = agent.train_model(model=model_a2c, 
#                              tb_log_name='a2c',
#                              total_timesteps=50000) if if_using_a2c else None
trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=args.total_timesteps) if if_using_a2c else None
'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion

# ### Agent 2: DDPG

# In[79]:


agent = DRLAgent(env = env_train)
model_ddpg = agent.get_model("ddpg")

if if_using_ddpg:
  # set up logger
  tmp_path = RESULTS_DIR + '/ddpg'
  new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_ddpg.set_logger(new_logger_ddpg)


# In[80]:

# region 修改为从入参导入
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''
# trained_ddpg = agent.train_model(model=model_ddpg, 
#                              tb_log_name='ddpg',
#                              total_timesteps=50000) if if_using_ddpg else None
trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=args.total_timesteps) if if_using_ddpg else None
'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion

# ### Agent 3: PPO

# In[81]:


agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

if if_using_ppo:
  # set up logger
  tmp_path = RESULTS_DIR + '/ppo'
  new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_ppo.set_logger(new_logger_ppo)


# In[82]:

# region 修改为从入参导入
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''
# trained_ppo = agent.train_model(model=model_ppo, 
#                              tb_log_name='ppo',
#                              total_timesteps=50000) if if_using_ppo else None
trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=args.total_timesteps) if if_using_ppo else None
'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion

# ### Agent 4: TD3

# In[83]:


agent = DRLAgent(env = env_train)
TD3_PARAMS = {"batch_size": 100, 
              "buffer_size": 1000000, 
              "learning_rate": 0.001}

model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)

if if_using_td3:
  # set up logger
  tmp_path = RESULTS_DIR + '/td3'
  new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_td3.set_logger(new_logger_td3)


# In[84]:

# region 修改为从入参导入
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''

# trained_td3 = agent.train_model(model=model_td3, 
#                              tb_log_name='td3',
#                              total_timesteps=30000) if if_using_td3 else None
trained_td3 = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=args.total_timesteps) if if_using_td3 else None

'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion

# ### Agent 5: SAC

# In[103]:


agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

if if_using_sac:
  # set up logger
  tmp_path = RESULTS_DIR + '/sac'
  new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_sac.set_logger(new_logger_sac)


# In[121]:

# region 修改为从入参导入
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''

# trained_sac = agent.train_model(model=model_sac, 
#                              tb_log_name='sac',
#                              total_timesteps=40000) if if_using_sac else None
trained_sac = agent.train_model(model=model_sac, 
                             tb_log_name='sac',
                             total_timesteps=args.total_timesteps) if if_using_sac else None

'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion


# ## In-sample Performance
# 
# Assume that the initial capital is $1,000,000.

# ### Set turbulence threshold
# Set the turbulence threshold to be greater than the maximum of insample turbulence data. If current turbulence index is greater than the threshold, then we assume that the current market is volatile

# In[122]:


data_risk_indicator = processed_full[(processed_full.date<TRAIN_END_DATE) & (processed_full.date>=TRAIN_START_DATE)]
insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])


# In[123]:


insample_risk_indicator.vix.describe()


# In[124]:


insample_risk_indicator.vix.quantile(0.996)


# In[125]:


insample_risk_indicator.turbulence.describe()


# In[126]:


insample_risk_indicator.turbulence.quantile(0.996)


# ### Trading (Out-of-sample Performance)
# 
# We update periodically in order to take full advantage of the data, e.g., retrain quarterly, monthly or weekly. We also tune the parameters along the way, in this notebook we use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends. 
# 
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

# In[127]:


e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()


# In[128]:


trade.head()


# In[129]:


trained_moedl = trained_sac
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_moedl, 
    environment = e_trade_gym)


# In[130]:


df_account_value.shape


# In[131]:


df_account_value.tail()


# In[132]:


df_actions.head()


# <a id='6'></a>
# # Part 7: Backtesting Results
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
# 

# In[133]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+RESULTS_DIR+"/perf_stats_all_"+now+'.csv')


# In[134]:


#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI", 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

stats = backtest_stats(baseline_df, value_col_name = 'close')


# In[135]:


df_account_value.loc[0,'date']


# In[136]:


df_account_value.loc[len(df_account_value)-1,'date']


# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# In[137]:


print("==============Compare to DJIA===========")
get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value, 
             baseline_ticker = '^DJI', 
             baseline_start = df_account_value.loc[0,'date'],
             baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])

#%%

# region 最后的输出和处理
'''
# Author: Edward
# ↓↓↓↓↓↓↓↓↓↓
'''
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.models import infer_signature

# 定义pyfunc模型
class ModelWrapper(PythonModel):

    def __init__(self, model,default_env_kwargs):
        self._model = model
        self._default_env_kwargs = default_env_kwargs
        
    def load_context(self, context):
        # self.model = mlflow.sklearn.load_model(context.artifacts["custom_model"])
        pass
    
    def predict_from(self, context: PythonModelContext, model_input, env_kwargs):
        '''
        预测行情，自定义环境，返回账户变动和买卖行为
        '''
        trade_gym = StockTradingEnv(df=model_input,**env_kwargs)
        return DRLAgent.DRL_prediction(
              model=self._model, 
              environment = trade_gym)
    def predict(self, context: PythonModelContext, model_input):
        '''
        预测行情，使用默认环境，只返回买卖行为
        '''
        trade_gym = StockTradingEnv(df=model_input, **self._default_env_kwargs)
        # trade_gym = StockTradingEnv(df = model_input.df, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
        # You don't have to keep the semantic meaning of `predict`. You can use here model.recommend(), model.forecast(), etc
        (df_account_value, df_actions) = DRLAgent.DRL_prediction(
            model=self._model, 
            environment = trade_gym)
        return df_actions
    # You can even add extra functions if you need to. Since the model is serialized,
    # all of them will be available when you load your model back.
    def predict_batch(self, data):
        pass




# 删除 Logger 以便序列化
trained_moedl.set_logger(None)
# 设置默认推理用环境参数
default_env_kwargs = {
  'turbulence_threshold' : 70,
  'risk_indicator_col':'vix',
  **env_kwargs
}
model_signature = infer_signature(trade,df_actions)
# 设置模型metadata
model_metadata = {
  'if_using_a2c' : if_using_a2c,
  'if_using_ddpg' : if_using_ddpg,
  'if_using_ppo' : if_using_ppo,
  'if_using_td3' : if_using_td3,
  'if_using_sac' : if_using_sac
}

# 导出df资源
mltool = tools.MLToolContext()
mltool.set_dir(RESULTS_DIR)

    # perf_stats_all.to_csv("./"+RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

mltool.to_csv(perf_stats_all,"perf_stats_all",index_label='Name', header=['Value'])
mltool.to_csv(df,"df",index_label='index')
mltool.to_csv(df_actions,"df_actions")
mltool.to_csv(df_account_value,"df_account_value",index_label="index")

# 导出 metric
mlflow.log_metric('train data length', len(train))
mlflow.log_metric('trade data length', len(trade))
for i in range(len(df_account_value)-1):
    # mlflow.log_metric('date',df_account_value.loc[i,'date'])
    mlflow.log_metric('account_value',df_account_value.loc[i,'account_value'])

# 输出模型和工件
mlflow.pyfunc.log_model(
   artifact_path='model',
   code_path=['Stock_NeurIPS2018_SB3_mlflow.py','tools.py','private_functions.py'],
   python_model=ModelWrapper(trained_moedl,default_env_kwargs),
   signature = model_signature,
   metadata = model_metadata)
mlflow.log_artifacts(RESULTS_DIR)

# 如果是ipython运行，结束MLflow上下文
if(not is_mlflow_project_mode):
  mlflow.end_run()

'''
加载模型样例程序
#model_output_sample = df_actions
'''
'''
加载模型和unwrap样例程序
#model_output_sample = (df_account_value, df_actions)
'''

#清理临时文件
import shutil
shutil.rmtree("./"+RESULTS_DIR)
print("==============Execution Finished===========")

# 关闭文件并将 stdout 恢复为默认值
sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr.close()
sys.stderr = sys.__stderr__

'''
# ↑↑↑↑↑↑↑↑↑↑
'''
#endregion