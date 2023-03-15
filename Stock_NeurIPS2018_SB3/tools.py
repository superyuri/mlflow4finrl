import mlflow
import os
import sys

def init_stdout():
    """
    拦截输出到MLProject Artifacts路径
    """
    RESULTS_DIR = 'results'
    os.makedirs('./'+RESULTS_DIR+'/logs', exist_ok=True)
    sys.stdout = open('./'+RESULTS_DIR+'/logs/log-std.log', 'w+')
    sys.stderr = open('./'+RESULTS_DIR+'/logs/log-err.log', 'w+')

def is_mlflow_project_mode():
    """

    :param eval_df:
    :return:

        True mlflow_project模式 

        False ipython模式
    """
    # run = mlflow.active_run()
    # if(run == None):
    #     return False
    # else:
    #     return True
    return True

import pathlib
import os


class MLToolContext:
    DIR = "."
    def set_dir(self,value):
        self.DIR = pathlib.Path(value).resolve()
    def to_csv(self,dataframe,filename,**kwargs):
        dataframe.to_csv(os.path.join(self.DIR,'logs',filename+'.csv'),**kwargs)