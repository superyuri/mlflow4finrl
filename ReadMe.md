# Mlflow Project for Finrl

## Prerequirements

* MLflow

    https://mlflow.org/docs/latest/quickstart.html

## File Structure
The main folder mlflow4finrl has several subfolders contains mlflow projects.



    mlflow4finrl
    |- Readme.md ............................ This file
    |- sample-minst ......................... Project folder : Minst Sample
        |- config ........................... Configs
            |- conda_environment.yaml ....... Conda config
        |- finrl_unsupported_artifact.py .... Source file
        |- MLProject......................... Mlflow project file
    |- sample-minst-customenv ............... Project folder : Minst Sample for custom env
    |- Stock_NeurIPS2018_SB3 ................ Project folder : Stock_NeurIPS2018_SB3 Implemetion
        |- config ........................... Configs
            |- conda_environment_fix.yaml ... Conda config
            |- conda_environment.yaml ....... Conda config
        |- notebook ......................... origin files
            |- Stock_NeurIPS2018_SB3.ipynb .. origin notebook
            |- Stock_NeurIPS2018_SB3.py ..... generated from Stock_NeurIPS2018_SB3.ipynb
        |- Stock_NeurIPS2018_SB3_mlflow.py .. main entry file, edit from Stock_NeurIPS2018_SB3.py
        |- tools.py ......................... useful tools for Mlflow project
        |- private_functions.py ............. private useful tools for current project
        |- MLProject......................... Mlflow project file
    ...

## Run a Mlflow project

example: run Stock_NeurIPS2018_SB3 for custom env from github

        mlflow run git@github.com:superyuri/mlflow4finrl.git#Stock_NeurIPS2018_SB3
    
example: run Stock_NeurIPS2018_SB3 from source

        mlflow run ./Stock_NeurIPS2018_SB3

example: run Minst Sample from github

        mlflow run git@github.com:superyuri/mlflow4finrl.git#sample-minst

example: run Minst Sample for custom env from github

        mlflow run git@github.com:superyuri/mlflow4finrl.git#sample-minst-customenv
        

## TODO 

1. mlflow implementation of finrl project

## Docs

* MLflow Projects

    https://mlflow.org/docs/latest/projects.html

清理环境

    for i in `conda env list|awk '{print $1}'|egrep -v 'base|#'|tr '\n' ' '`;do echo $i;conda env remove --name $i;done
