import warnings
import numpy as np
import pandas as pd

from env_multiple_crypto import CryptoEnv
from env_advance_crypto import AdvCryptoEnv
from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.meta.data_processor import DataProcessor
import gym
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

#TICKER_LIST = ['BTC-USD','ETH-USD','ADA-USD','BNB-USD','XRP-USD',
#                'SOL-USD','DOT-USD', 'DOGE-USD','AVAX-USD','UNI-USD']
#TICKER_LIST = ['BTC-JPY','ETH-JPY','BCH-JPY','LTC-JPY','XRP-JPY', 'XEM-JPY','XLM-JPY', 'BAT-JPY','OMG-JPY','XTZ-JPY']
TICKER_LIST = ['BTC','ETH','BCH','LTC','XRP', 'XEM','XLM']
INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}

DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": 0.1,
}
DQN_PARAMS = {
    "learning_rate": 0.01,
    "reward_decay": 0.9,
    "e_greedy": 0.9,
    "replace_target_iter": 300,
    "memory_size": 500,
    "batch_size": 32,
    "e_greedy_increment": None,
}

class CryptoAll:

    def __init__(self, 
                fl_model_name
                ):

        #get data
        self.fl_model_name = fl_model_name

        warnings.simplefilter('ignore')

    def train(self,start_date, end_date, ticker_list, data_source, time_interval, 
            technical_indicator_list, drl_lib, env, model_name, if_vix=True,
            **kwargs):
        
        #process data using unified data processor
        DP = DataProcessor(data_source, **kwargs)
        DP.setPara(ticker_list,
                    start_date,
                    end_date, 
                    time_interval)
        downloadData = DP.download_data()
        '''downloadData = pd.read_csv("all.csv", names=('date','tic','open', 'high', 'low','close','volume','adjcp','day'), skiprows=1)
        try:
            # convert the column names to standardized names
            downloadData.columns = [
                "date",
                "tic",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjcp",
                "day",
            ]
        except NotImplementedError:
            print("the features are not supported currently")'''
        data = DP.clean_data(downloadData)
        data.to_csv("all_clean.csv")
        data = DP.add_technical_indicator(data, technical_indicator_list)
        data.to_csv("all_technical_indicator.csv")
        data = DP.add_turbulence(data)
        data.to_csv("all_turbulence.csv")
        if if_vix:
            data = DP.add_vix(data)
        data.to_csv("all_vix.csv")
        
        date_array,high_array,low_array, price_array, tech_array, turbulence_array = DP.df_to_array_new(data,if_vix)
        data_config = {'date_array': date_array,
                        'high_array':high_array,
                        'low_array':low_array,
                    'price_array': price_array,
                    'tech_array': tech_array,
                    'turbulence_array': turbulence_array}

        #build environment using processed data
        if(self.fl_model_name == 'multiple'):
            env = CryptoEnv
            env_instance = env(config=data_config)
        elif(self.fl_model_name == 'advance'):
            env = AdvCryptoEnv
            env_instance = env('data',37,505,data_config,1,1000000,0.01,0.01,0.99,None,False,True,'P',model_name,False,False)
        else:
            raise ValueError("env is NOT supported. Please check.")

        #read parameters and load agents
        current_working_dir = kwargs.get('current_working_dir','./modal/' + self.fl_model_name+"_"+str(model_name))

        if drl_lib == 'elegantrl':
            break_step = kwargs.get('break_step', 1e6)
            erl_params = kwargs.get('erl_params')

            agent = DRLAgent_erl(env = env,
                                price_array = price_array,
                                tech_array=tech_array,
                                turbulence_array=turbulence_array)

            model = agent.get_model(model_name, model_kwargs = erl_params)

            trained_model = agent.train_model(model=model, 
                                            cwd=current_working_dir,
                                            total_timesteps=break_step)
            
        
        elif drl_lib == 'stable_baselines3':
            total_timesteps = kwargs.get('total_timesteps', 1e5)
            agent_params = kwargs.get('agent_params')

            agent = DRLAgent_sb3(env = env_instance)

            model = agent.get_model(model_name, model_kwargs = agent_params)
            trained_model = agent.train_model(model=model, 
                                    tb_log_name=model_name,
                                    total_timesteps=total_timesteps)
            print('Training finished!')
            trained_model.save(current_working_dir)
            print('Trained model saved in ' + str(current_working_dir))
        else:
            raise ValueError('DRL library input is NOT supported. Please check.')

    def test(self,start_date, end_date, ticker_list, data_source, time_interval,
                technical_indicator_list, drl_lib, env, model_name, if_vix=True,
                **kwargs):
    
        #process data using unified data processor
        DP = DataProcessor(data_source, **kwargs)
        DP.setPara(ticker_list,
                    start_date,
                    end_date, 
                    time_interval)
        downloadData = DP.download_data()
        data = DP.clean_data(downloadData)
        data = DP.add_technical_indicator(data, technical_indicator_list)
        data = DP.add_turbulence(data)
        if if_vix:
            data = DP.add_vix(data)
        
        date_array,high_array,low_array, price_array, tech_array, turbulence_array = DP.df_to_array_new(data,if_vix)
        data_config = {'date_array': date_array,
                        'high_array':high_array,
                        'low_array':low_array,
                    'price_array': price_array,
                    'tech_array': tech_array,
                    'turbulence_array': turbulence_array}
            
        np.save('./price_array.npy', price_array)
        #build environment using processed data
        if(self.fl_model_name == 'multiple'):
            env = CryptoEnv
            env_instance = env(config=data_config)
        elif(self.fl_model_name == 'advance'):
            env = AdvCryptoEnv
            env_instance = env('data',37,505,data_config,1,1000000,0.01,0.01,0.99,None,True,True,'P',model_name,True,False)
        else:
            raise ValueError("env is NOT supported. Please check.")


        # load elegantrl needs state dim, action dim and net dim
        net_dimension = kwargs.get("net_dimension", 2 ** 7)
        current_working_dir = kwargs.get("current_working_dir", "./modal/" + self.fl_model_name+"_"+str(model_name))
        print("price_array: ", len(price_array))

        if drl_lib == "elegantrl":
            episode_total_assets = DRLAgent_erl.DRL_prediction(
                model_name=model_name,
                cwd=current_working_dir,
                net_dimension=net_dimension,
                environment=env_instance,
            )

            return episode_total_assets

        elif drl_lib == "stable_baselines3":
            episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
                model_name=model_name, environment=env_instance, cwd=current_working_dir
            )

            return episode_total_assets
        else:
            raise ValueError("DRL library input is NOT supported. Please check.")

    def make_plot(self, account_value_erl, path,rl_model_name):
        account_value_erl = np.array(account_value_erl)
        agent_returns = account_value_erl/account_value_erl[0]
        #calculate buy-and-hold btc returns
        price_array = np.load('./price_array.npy')
        btc_prices = price_array[:,0]
        buy_hold_btc_returns = btc_prices/btc_prices[0]
        #calculate equal weight portfolio returns
        price_array = np.load('./price_array.npy')
        initial_prices = price_array[0,:]
        equal_weight = np.array([1e5/initial_prices[i] for i in range(len(TICKER_LIST))])
        equal_weight_values = []
        for i in range(0, price_array.shape[0]):
            equal_weight_values.append(np.sum(equal_weight * price_array[i]))
        equal_weight_values = np.array(equal_weight_values)
        equal_returns = equal_weight_values/equal_weight_values[0]
        #plot 
        plt.figure(dpi=200)
        plt.grid()
        plt.grid(which='minor', axis='y')
        plt.title('Cryptocurrency Trading ', fontsize=20)
        plt.plot(agent_returns, label='ElegantRL Agent', color = 'red')
        plt.plot(buy_hold_btc_returns, label='Buy-and-Hold BTC', color='blue')
        plt.plot(equal_returns, label='Equal Weight Portfolio', color='green')
        plt.ylabel('Return', fontsize=16)
        plt.xlabel('Times (5min)', fontsize=16)
        plt.xticks(size=14)
        plt.yticks(size=14)
        '''ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(210))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(21))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([]))'''
        plt.legend(fontsize=10.5)
        fileName = './' + path +'/'+self.fl_model_name+"_"+ rl_model_name + '.png'
        plt.savefig(fileName)
        plt.close()
    
# 動作確認
if __name__ == '__main__':
    TRAIN_START_DATE = '2022-07-01'
    TRAIN_END_DATE = '2022-08-31'

    TEST_START_DATE = '2022-09-01'
    TEST_END_DATE = '2022-09-30'

    DRL_LIB = 'stable_baselines3' #'elegantrl','stable_baselines3'
    API_KEY = "1ddcbec72bef777aaee9343272ec1467"
    API_SECRET = "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf"
    API_BASE_URL = "https://paper-api.alpaca.markets"
    
    DATA_SOURCE='gmo'#'yahoofinance','gmo'
    TIME_INTERVAL='1Min'#'1D','1Min'

    #fl_model_names = ['multiple','advance']
    fl_model_names = ['advance']
    #rl_model_names = ['A2C','DDPG','PPO','SAC','TD3','DQN']
    rl_model_names = ['A2C']
    for fl_model_name in fl_model_names:
        if(fl_model_name == 'multiple'):
            env = CryptoEnv
        elif(fl_model_name == 'advance'):
            env = AdvCryptoEnv
        else:
            raise ValueError("env is NOT supported. Please check.")
        
        for rl_model_name in rl_model_names:
            CURRENT_WORKING_DIR = './modal/'+fl_model_name+"_"+rl_model_name.lower()
            if(rl_model_name == 'A2C'):
                env_kwargs = {
                    "API_KEY": API_KEY, 
                    "API_SECRET": API_SECRET, 
                    "API_BASE_URL": API_BASE_URL,
                    "rllib_params": RLlib_PARAMS,
                    "agent_params": A2C_PARAMS,
                }
            elif(rl_model_name == 'DDPG'):
                env_kwargs = {
                    "API_KEY": API_KEY, 
                    "API_SECRET": API_SECRET, 
                    "API_BASE_URL": API_BASE_URL,
                    "rllib_params": RLlib_PARAMS,
                    "agent_params": DDPG_PARAMS,
                }
            elif(rl_model_name == 'PPO'):
                env_kwargs = {
                    "API_KEY": API_KEY, 
                    "API_SECRET": API_SECRET, 
                    "API_BASE_URL": API_BASE_URL,
                    "rllib_params": RLlib_PARAMS,
                    "agent_params": PPO_PARAMS,
                }
            elif(rl_model_name == 'SAC'):
                env_kwargs = {
                    "API_KEY": API_KEY, 
                    "API_SECRET": API_SECRET, 
                    "API_BASE_URL": API_BASE_URL,
                    "rllib_params": RLlib_PARAMS,
                    "agent_params": SAC_PARAMS,
                }
            elif(rl_model_name == 'TD3'):
                env_kwargs = {
                    "API_KEY": API_KEY, 
                    "API_SECRET": API_SECRET, 
                    "API_BASE_URL": API_BASE_URL,
                    "rllib_params": RLlib_PARAMS,
                    "agent_params": TD3_PARAMS,
                }
            elif(rl_model_name == 'DQN'):
                env_kwargs = {
                    "API_KEY": API_KEY, 
                    "API_SECRET": API_SECRET, 
                    "API_BASE_URL": API_BASE_URL,
                    "rllib_params": RLlib_PARAMS,
                    "agent_params": DQN_PARAMS,
                }
            else:
                raise ValueError("rl_model is NOT supported. Please check.")

            #レース訓練
            cryptoAll = CryptoAll(fl_model_name)
            #ポリシー訓練
            cryptoAll.train(start_date=TRAIN_START_DATE, 
                end_date=TRAIN_END_DATE,
                ticker_list=TICKER_LIST, 
                data_source=DATA_SOURCE,
                time_interval=TIME_INTERVAL, 
                technical_indicator_list=INDICATORS,
                drl_lib=DRL_LIB, 
                env=env, 
                model_name=rl_model_name.lower(), 
                current_working_dir=CURRENT_WORKING_DIR,
                erl_params=ERL_PARAMS,
                break_step=5e4,
                if_vix=False,
                **env_kwargs
                )

            #ポリシー評価
            cryptoAll = CryptoAll(fl_model_name)
            account_value_erl = cryptoAll.test(start_date = TEST_START_DATE, 
                    end_date = TEST_END_DATE,
                    ticker_list = TICKER_LIST, 
                    data_source = DATA_SOURCE,
                    time_interval= TIME_INTERVAL, 
                    technical_indicator_list= INDICATORS,
                    drl_lib=DRL_LIB, 
                    env=env, 
                    model_name=rl_model_name.lower(), 
                    current_working_dir=CURRENT_WORKING_DIR, 
                    net_dimension = 2**9, 
                    if_vix=False
                    )
            cryptoAll.make_plot(account_value_erl,'data',rl_model_name.lower())                        