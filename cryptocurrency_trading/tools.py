import mlflow
import os
import sys

MLFLOW_RUN_ID = os.getenv('MLFLOW_RUN_ID')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT_ID = os.getenv('MLFLOW_EXPERIMENT_ID')
def init_stdout(cwd):
    """
    拦截输出到MLProject Artifacts路径
    """
    os.makedirs(os.path.join(cwd,'logs'), exist_ok=True)
    sys.stdout = open(os.path.join(cwd,'logs','log-std.log'), 'w+')
    sys.stderr = open(os.path.join(cwd,'logs','log-err.log'), 'w+')

def is_mlflow_project_mode():
    """

    :param eval_df:
    :return:

        True mlflow_project模式 

        False ipython模式
    """

    if(MLFLOW_RUN_ID is not None):
        return True
    else:
        return False
def get_active_run_if_mlflow_project_mode():
    """

    """
    if(MLFLOW_RUN_ID is not None):
        return mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI,MLFLOW_EXPERIMENT_ID).get_run(MLFLOW_RUN_ID)
    else:
        return None
import pathlib
import os


class MLToolContext:
    DIR = "."
    def set_dir(self,value):
        self.DIR = pathlib.Path(value).resolve()
    def to_csv(self,dataframe,filename,**kwargs):
        dataframe.to_csv(os.path.join(self.DIR,'logs',filename+'.csv'),**kwargs)