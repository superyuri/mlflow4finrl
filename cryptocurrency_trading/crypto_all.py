from typing import Type
import warnings
import numpy as np
import pandas as pd

from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
from finrl.meta.env_cryptocurrency_trading.env_advance_crypto import AdvCryptoEnv
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
from pathlib import Path
import os
class CryptoAgent:

    __DEFAULT_GYM_ENV_TYPE = 'multiple'
    __DEFAULT_CWD = './data'

    def __init__(self, cwd : str = __DEFAULT_CWD, is_output_artifacts : bool = True):
        self.is_output_artifacts = is_output_artifacts
        self.cwd = cwd
        print(self.__get_data_cwd())
        print(self.__get_plot_cwd())
        if self.is_output_artifacts:
            Path(self.__get_data_cwd()).mkdir(parents=True,exist_ok=True)
            Path(self.__get_plot_cwd()).mkdir(parents=True,exist_ok=True)
        warnings.simplefilter('ignore')

    def __get_data_cwd(self) -> str:
        return os.path.join(self.cwd,"data")
    
    def __get_plot_cwd(self) -> str:
        return os.path.join(self.cwd,"plot")
    
    def __output_df_to_csv(self, df: pd.DataFrame,filename: str,extension: str = ".csv"):
        if self.is_output_artifacts:
            df.to_csv(os.path.join(self.__get_data_cwd(),filename+extension))

    def __output_ndarray_to_npy(self, nd: np.ndarray ,filename: str,extension: str = ".npy"):
        if self.is_output_artifacts:
            fullname = os.path.join(self.__get_data_cwd(), filename+ extension)
            np.save(file = fullname, arr = nd)

    def __load_ndarray_from_npy(self,filename: str,extension: str = ".npy") -> np.ndarray:
        # if self.is_output_artifacts:
        return np.load(os.path.join(self.__get_data_cwd(),filename+extension))
    
    def __output_plot_to_img(self,filename: str,extension: str = ".png"):
        if self.is_output_artifacts:
            plt.savefig(os.path.join(self.__get_plot_cwd(),filename+extension))


    def __generate_gym_env_by_type(self, gym_env_type : str, model_name, data_config, is_output : bool= False) -> tuple[Type[gym.Env], gym.Env]:
        if(gym_env_type.lower() == 'multiple'):
            env = CryptoEnv
            env_instance = env(config=data_config)
        elif(gym_env_type.lower() == 'advance'):
            print("__generate_gym_env_by_type")
            print("path " + self.__get_data_cwd())
            env = AdvCryptoEnv
            env_instance = env(path = self.__get_data_cwd(),state_space = 37,action_space = 505,
                               config = data_config,lookback = 1,initial_amount = 1000000,
                               buy_cost_pct = 0.01,sell_cost_pct = 0.01,gamma = 0.99,
                               turbulence_threshold = None,make_plots = is_output,initial = True,
                               prefix = 'P', modal_name = model_name, make_csv = is_output, is_real = False)
        else:
            raise ValueError("env is NOT supported. Please check.")
        
        return env,env_instance

    def train(self,start_date, end_date, ticker_list, data_source, time_interval, 
            technical_indicator_list, drl_lib : str, model_name : str, gym_env_type : str = __DEFAULT_GYM_ENV_TYPE, 
            if_vix=True, model_file_fullname = None, **kwargs):
        
        drl_lib = drl_lib.lower()
        model_name = model_name.lower()
        gym_env_type = gym_env_type.lower()
    
        if model_file_fullname is None:
            model_file_fullname = "./" + str(gym_env_type) + '_'+ str(model_name)
        Path(model_file_fullname).parent.mkdir(parents=True,exist_ok=True)

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
        self.__output_df_to_csv(data,"all_clean")
        data = DP.add_technical_indicator(data, technical_indicator_list)
        self.__output_df_to_csv(data,"all_technical_indicator")
        data = DP.add_turbulence(data)
        self.__output_df_to_csv(data,"all_turbulence")
        if if_vix:
            data = DP.add_vix(data)
        self.__output_df_to_csv(data,"all_vix")
        
        date_array,high_array,low_array, price_array, tech_array, turbulence_array = DP.df_to_array_new(data,if_vix)
        data_config = {'date_array': date_array,
                        'high_array':high_array,
                        'low_array':low_array,
                    'price_array': price_array,
                    'tech_array': tech_array,
                    'turbulence_array': turbulence_array}

        #build environment using processed data
        env,env_instance = self.__generate_gym_env_by_type(gym_env_type, model_name, data_config,is_output=False)


        #read parameters and load agents
        #current_working_dir = #kwargs.get('current_working_dir','./model/' + self.fl_model_name+"_"+str(model_name))
        if drl_lib == 'elegantrl':
            break_step = kwargs.get('break_step', 1e6)
            erl_params = kwargs.get('erl_params')

            agent = DRLAgent_erl(env = env,
                                price_array = price_array,
                                tech_array=tech_array,
                                turbulence_array=turbulence_array)

            model = agent.get_model(model_name, model_kwargs = erl_params)

            trained_model = agent.train_model(model=model, 
                                            cwd=model_file_fullname,
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
            trained_model.save(model_file_fullname)
            print('Trained model saved in ' + model_file_fullname)
        else:
            raise ValueError('DRL library input is NOT supported. Please check.')

    def test(self,start_date, end_date, ticker_list, data_source, time_interval,
                technical_indicator_list, drl_lib : str, model_name : str, gym_env_type : str = __DEFAULT_GYM_ENV_TYPE, 
                if_vix=True, model_file_fullname = None, **kwargs):
    
        drl_lib = drl_lib.lower()
        model_name = model_name.lower()
        gym_env_type = gym_env_type.lower()

        if model_file_fullname is None:
            model_file_fullname = "./" + gym_env_type + '_'+ model_name
        #Path(model_file_fullname).parent.mkdir(parents=True,exist_ok=True)

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
        #np.save(os.path.join(self.__get_data_cwd(),'price_array.npy'), price_array)
        self.__output_ndarray_to_npy(price_array,"price_array")
        # np.save('./.npy', price_array)
        #build environment using processed data
        env,env_instance = self.__generate_gym_env_by_type(gym_env_type, model_name, data_config,is_output=True)

        # load elegantrl needs state dim, action dim and net dim
        net_dimension = kwargs.get("net_dimension", 2 ** 7)

        print("price_array: ", len(price_array))

        if drl_lib == "elegantrl":
            episode_total_assets = DRLAgent_erl.DRL_prediction(
                model_name=model_name,
                cwd=model_file_fullname,
                net_dimension=net_dimension,
                environment=env_instance,
            )

            return episode_total_assets

        elif drl_lib == "stable_baselines3":
            episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
                model_name=model_name, environment=env_instance, cwd=model_file_fullname
            )

            return episode_total_assets
        else:
            raise ValueError("DRL library input is NOT supported. Please check.")
    def get_data_config(self, start_date, end_date, ticker_list, data_source, time_interval,
                technical_indicator_list,if_vix=True,**kwargs):

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
        return data_config
    
    def predict(self, data_config, drl_lib : str, model_name : str, gym_env_type : str, 
                if_vix=True, model_file_fullname = None, **kwargs):
    
        drl_lib = drl_lib.lower()
        model_name = model_name.lower()
        gym_env_type = gym_env_type.lower()
        price_array = data_config['price_array']

        if model_file_fullname is None:
            raise ValueError("model_file_fullname is wrong. Please check.")

        #build environment using processed data
        env,env_instance = self.__generate_gym_env_by_type(gym_env_type, model_name, data_config,is_output=True)

        # load elegantrl needs state dim, action dim and net dim
        net_dimension = kwargs.get("net_dimension", 2 ** 7)

        print("price_array: ", len(price_array))

        if drl_lib == "elegantrl":
            episode_total_assets = DRLAgent_erl.DRL_prediction(
                model_name=model_name,
                cwd=model_file_fullname,
                net_dimension=net_dimension,
                environment=env_instance,
            )

            return episode_total_assets

        elif drl_lib == "stable_baselines3":
            episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
                model_name=model_name, environment=env_instance, cwd=model_file_fullname
            )

            return episode_total_assets
        else:
            raise ValueError("DRL library input is NOT supported. Please check.")
        

    def make_sample_plot(self, drl_lib : str, account_value_erl, filename):
        account_value_erl = np.array(account_value_erl)
        agent_returns = account_value_erl/account_value_erl[0]
        #calculate buy-and-hold btc returns
        price_array = self.__load_ndarray_from_npy('price_array')
        btc_prices = price_array[:,0]
        buy_hold_btc_returns = btc_prices/btc_prices[0]
        #calculate equal weight portfolio returns
        price_array = self.__load_ndarray_from_npy('price_array')
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
        plt.plot(agent_returns, label = drl_lib+' Agent', color = 'red')
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
        self.__output_plot_to_img(filename=filename)
        plt.close()

    @classmethod
    def get_default_env_kwargs(self, model_name : str, **kwargs):
        if(model_name.upper() == 'A2C'):
            return {
                "API_KEY": kwargs.get('api_key'),
                "API_SECRET": kwargs.get('api_secret'),
                "API_BASE_URL": kwargs.get('api_base_url'),
                "rllib_params": RLlib_PARAMS,
                "agent_params": A2C_PARAMS,
            }
        elif(model_name.upper() == 'DDPG'):
            return {
                "API_KEY": kwargs.get('api_key'),
                "API_SECRET": kwargs.get('api_secret'),
                "API_BASE_URL": kwargs.get('api_base_url'),
                "rllib_params": RLlib_PARAMS,
                "agent_params": DDPG_PARAMS,
            }
        elif(model_name.upper() == 'PPO'):
            return {
                "API_KEY": kwargs.get('api_key'),
                "API_SECRET": kwargs.get('api_secret'),
                "API_BASE_URL": kwargs.get('api_base_url'),
                "rllib_params": RLlib_PARAMS,
                "agent_params": PPO_PARAMS,
            }
        elif(model_name.upper() == 'SAC'):
            return {
                "API_KEY": kwargs.get('api_key'),
                "API_SECRET": kwargs.get('api_secret'),
                "API_BASE_URL": kwargs.get('api_base_url'),
                "rllib_params": RLlib_PARAMS,
                "agent_params": SAC_PARAMS,
            }
        elif(model_name.upper() == 'TD3'):
            return {
                "API_KEY": kwargs.get('api_key'),
                "API_SECRET": kwargs.get('api_secret'),
                "API_BASE_URL": kwargs.get('api_base_url'),
                "rllib_params": RLlib_PARAMS,
                "agent_params": TD3_PARAMS,
            }
        elif(model_name.upper() == 'DQN'):
            return {
                "API_KEY": kwargs.get('api_key'),
                "API_SECRET": kwargs.get('api_secret'),
                "API_BASE_URL": kwargs.get('api_base_url'),
                "rllib_params": RLlib_PARAMS,
                "agent_params": DQN_PARAMS,
            }
        else:
            raise ValueError("model is NOT supported. Please check.")