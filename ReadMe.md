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
    |- sample-minst-customenv ......................... Project folder : Minst Sample for custom env
    |- finrl ................................ Project folder : Finrl Implemetion
    ...

## Run a Mlflow project

example: run Minst Sample from source

        mlflow run ./sample-minst

example: run Minst Sample from github

        mlflow run mlflow run git@github.com:superyuri/mlflow4finrl.git#sample-minst

example: run Minst Sample for custom env from github

        mlflow run mlflow run git@github.com:superyuri/mlflow4finrl.git#sample-minst-customenv


## TODO 

1. mlflow implementation of finrl project

## Docs

* MLflow Projects

    https://mlflow.org/docs/latest/projects.html