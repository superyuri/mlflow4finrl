import os 
from crypto_all import CryptoAgent

def excute_start(args, current_working_dir = None):
    if current_working_dir is None:
        current_working_dir = os.path(os.getcwd())
    print("CURRENT_WORKING_DIR:")
    print(current_working_dir)

    import mlflow
    import sys

    import crypto_all
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
    from crypto_all import TICKER_LIST,INDICATORS,ERL_PARAMS,RLlib_PARAMS,PPO_PARAMS,A2C_PARAMS,DDPG_PARAMS,TD3_PARAMS,SAC_PARAMS,DQN_PARAMS



    TRAIN_START_DATE = args.train_start_date
    TRAIN_END_DATE =  args.train_end_date

    TEST_START_DATE = args.test_start_date
    TEST_END_DATE = args.test_end_date

    DRL_LIB = 'stable_baselines3' #'elegantrl','stable_baselines3'
    API_KEY = "1ddcbec72bef777aaee9343272ec1467"
    API_SECRET = "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf"
    API_BASE_URL = "https://paper-api.alpaca.markets"

    DATA_SOURCE='gmo'#'yahoofinance','gmo'
    TIME_INTERVAL='1Min'#'1D','1Min'

    gym_env_types = ['multiple','advance']
    # gym_env_types = ['multiple']
    # gym_env_types = ['advance']
    # /multiple_ddpg
    # AttributeError: 'NoneType' object has no attribute 'get'
    # /advance_ddpg
    # AttributeError: 'NoneType' object has no attribute 'get'
    # /multiple_ppo
    # OK
    # /advance_ppo
    # OK
    # /multiple_sac
    # AttributeError: 'NoneType' object has no attribute 'get'
    # /advance_sac
    # AttributeError: 'NoneType' object has no attribute 'get'
    # /multiple_td3
    # AttributeError: 'NoneType' object has no attribute 'get'
    # advance_td3
    # AttributeError: 'NoneType' object has no attribute 'get'
    # /multiple_dqn
    # NotImplementedError: NotImplementedError
    # /advance_dqn
    # NameError: name 'traceback' is not defined
    #model_names = ['A2C','DDPG','PPO','SAC','TD3','DQN']
    model_names = ['A2C','PPO']
    #model_names = ['A2C']
    for gym_env_type in gym_env_types:
        for model_name in model_names:

            tags = {
                'model_name':model_name,
                'gym_env_type':gym_env_type
            }
            model_filename = gym_env_type.lower()+"_"+model_name.lower()
            model_file_fullname = os.path.join(current_working_dir,model_filename+".zip") 

            with mlflow.start_run(run_name = model_filename, nested= True,tags=tags) as train_run:
                
                print(f"==============Mlflow Task: {train_run.info.run_id} Start===========")
                # 输出参数
                # for key, value in vars(args).items():
                #     mlflow.log_param(key, value)
                mlflow.log_param('model_name',model_name)
                mlflow.log_param('gym_env_type',gym_env_type)
                mlflow.log_param('del_lib',DRL_LIB)
                mlflow.log_param('data_source',DATA_SOURCE)
                mlflow.log_param('time_interval',TIME_INTERVAL)

                #CURRENT_WORKING_DIR = './modal/'+fl_model_name+"_"+rl_model_name.lower()
                print("train "+model_filename)

                env_kwargs = CryptoAgent.get_default_env_kwargs(
                    model_name = model_name, api_key = API_KEY, api_secret = API_SECRET, api_base_url = API_BASE_URL)
                
                env_kwargs['total_timesteps'] = args.total_timesteps

                #レース訓練
                cryptoAgent = CryptoAgent(cwd = current_working_dir, is_output_artifacts = True)

                print("==============Training Start===========")
                #ポリシー訓練
                cryptoAgent.train(start_date=TRAIN_START_DATE,
                    end_date=TRAIN_END_DATE,
                    ticker_list=TICKER_LIST,
                    data_source=DATA_SOURCE,
                    time_interval=TIME_INTERVAL,
                    technical_indicator_list=INDICATORS,
                    drl_lib=DRL_LIB,
                    model_name=model_name,
                    gym_env_type = gym_env_type,
                    if_vix=False,
                    model_file_fullname=model_file_fullname,
                    erl_params=ERL_PARAMS,
                    break_step=5e4,
                    **env_kwargs
                    )
                # 输出模型
                
                print("==============Training Finished, output mlflow pyfunc model===========")

                mlflow.pyfunc.log_model(
                    artifact_path='model',
                    #artifact_path="artifacts",
                    code_path=['main.py','crypto_all.py','tools.py','private_functions.py'],
                    artifacts={"custom_model": model_file_fullname},
                    python_model=ModelWrapper(drl_lib=DRL_LIB, 
                                              model_name= model_name, 
                                              gym_env_type= gym_env_type,
                                              if_vix= False,
                                              **env_kwargs
                                              ),
                    # signature = model_signature,
                    # metadata = model_metadata,
                )
                
                print("==============Testing Start===========")
                #ポリシー評価
                cryptoAgent = CryptoAgent(cwd = current_working_dir, is_output_artifacts = True)
                account_value_erl = cryptoAgent.test(start_date = TEST_START_DATE,
                        end_date = TEST_END_DATE,
                        ticker_list = TICKER_LIST,
                        data_source = DATA_SOURCE,
                        time_interval= TIME_INTERVAL,
                        technical_indicator_list= INDICATORS,
                        drl_lib=DRL_LIB,
                        model_name=model_name,
                        gym_env_type = gym_env_type,
                        if_vix=False,
                        model_file_fullname=model_file_fullname,
                        net_dimension = 2**9,
                        )
                print("==============Testing Finished,output plots===========")
                cryptoAgent.make_sample_plot(DRL_LIB, account_value_erl,model_filename)

                print("==============load mlflow pyfunc model===========")
                
                mlflow_model_info = f'runs:/{train_run.info.run_id}/model'

                # Load model as a PyFuncModel.
                mlflow_model = mlflow.pyfunc.load_model(mlflow_model_info)

                print("==============load data for predict===========")

                get_data_config = cryptoAgent.get_data_config(start_date = TEST_START_DATE, end_date = TEST_END_DATE, ticker_list = TICKER_LIST, data_source = DATA_SOURCE, time_interval = TIME_INTERVAL,
                technical_indicator_list = INDICATORS, if_vix=False,**env_kwargs)
                
                
                print("==============Predicting Start===========")
                result = mlflow_model.predict(get_data_config)

                print("==============Predicting Finished===========")

                print('loaded_model outputs:')
                for i in range(len(result)-1):
                    # mlflow.log_metric('date',df_account_value.loc[i,'date'])
                    mlflow.log_metric('account_value',result[i])
                # output plot
                # mlflow.log_artifact(os.path.join(cryptoAgent.__get_plot_cwd()+tempDir.path(),model_filename+".png"))
                
                print(f"==============Mlflow Task: {train_run.info.run_id} Finished===========")


from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.models import infer_signature

# 定义pyfunc模型
class ModelWrapper(PythonModel,):

    def __init__(self, drl_lib : str, model_name : str, gym_env_type : str, 
                if_vix=True, **kwargs) -> None:
        super().__init__()
        self.drl_lib = drl_lib
        self.model_name = model_name
        self.gym_env_type = gym_env_type
        self.if_vix = if_vix
        self.kwargs = kwargs

    def load_context(self, context: PythonModelContext):
        self.model_file_fullname = context.artifacts["custom_model"]
        print("def load_context(self, context):context.artifacts[custom_model]")
        print(self.model_file_fullname)
        # self.model = mlflow.sklearn.load_model()
        #print('self.model')
        #print(self.model)

    def predict(self, context: PythonModelContext, model_input):
        with TempDir() as tempDir:
            return CryptoAgent(cwd = tempDir.path()).predict(
                model_input,
                self.drl_lib,
                self.model_name,
                self.gym_env_type,
                self.if_vix,
                self.model_file_fullname,
                **self.kwargs,
                )


    # 设置默认推理用环境参数
    # default_env_kwargs = {
    #   'turbulence_threshold' : 70,
    #   'risk_indicator_col':'vix',
    #   **env_kwargs
    # }
    # model_signature = infer_signature(trade,df_actions)
    # 设置模型metadata
    # model_metadata = {
    #   'if_using_a2c' : if_using_a2c,
    #   'if_using_ddpg' : if_using_ddpg,
    #   'if_using_ppo' : if_using_ppo,
    #   'if_using_td3' : if_using_td3,
    #   'if_using_sac' : if_using_sac
    # }

    # 导出df资源
    # mltool = tools.MLToolContext()
    # RESULTS_DIR ='results'
    # mltool.set_dir(RESULTS_DIR)

        # perf_stats_all.to_csv("./"+RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

    # mltool.to_csv(perf_stats_all,"perf_stats_all",index_label='Name', header=['Value'])
    # mltool.to_csv(df,"df",index_label='index')
    # mltool.to_csv(df_actions,"df_actions")
    # mltool.to_csv(df_account_value,"df_account_value",index_label="index")

    # 导出 metric
    # mlflow.log_metric('train data length', len(train))
    # mlflow.log_metric('trade data length', len(trade))
    # for i in range(len(df_account_value)-1):
    #     # mlflow.log_metric('date',df_account_value.loc[i,'date'])
    #     mlflow.log_metric('account_value',df_account_value.loc[i,'account_value'])


    '''
    加载模型样例程序
    #model_output_sample = df_actions
    '''
    '''
    加载模型和unwrap样例程序
    #model_output_sample = (df_account_value, df_actions)
    '''



if __name__ == '__main__':
    import mlflow
    import private_functions
    import tools
    import sys
    from mlflow.utils.file_utils import TempDir, path_to_local_file_uri
    from mlflow import tracking

    # 初始化参数
    args = private_functions.init_args()

    
    with TempDir() as tempDir:

        # 拦截文本输出
        tools.init_stdout(tempDir.path())

        with mlflow.start_run() as main_run:

            if(tools.is_mlflow_project_mode()):
                print("WORKING MODE: MLFLOW PROJECT")
                # main_run = tools.get_active_run_if_mlflow_project_mode()
            else:
                # 如果是python运行，创建MLflow上下文
                print("WORKING MODE: PYTHON")
                # 输出参数
                for key, value in vars(args).items():
                    mlflow.log_param(key, value)

            print(main_run.info)

            # with TempDir() as tmp:
            #     CURRENT_WORKING_DIR = tmp.path()
            #     print("CURRENT_WORKING_DIR:"+CURRENT_WORKING_DIR)

                # data_path = tmp.path("image_model")
                # os.mkdir(data_path)
                #os.mkdir(temp_path)

            try:
                excute_start(args, tempDir.path())
            except Exception as e: 

                sys.stderr.write(traceback.format_exc())
                
                # 关闭文件并将 stdout 恢复为默认值
                sys.stdout.close()
                sys.stdout = sys.__stdout__
                sys.stderr.close()
                sys.stderr = sys.__stderr__
                
                raise e
            finally:
                # 输出模型和工件

                mlflow.log_artifacts(tempDir.path())
            
                #清理临时文件
                #import shutil
                #shutil.rmtree("./"+RESULTS_DIR)
                #$print("==============Execution Finished===========")

