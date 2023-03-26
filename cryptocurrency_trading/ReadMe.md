# cryptocurrency_trading

## 简介

本项目是由[crypto_all.py](https://github.com/highwayns/FinRL/blob/master/finrl/meta/env_cryptocurrency_trading/crypto_all.py)的项目修改而来。
目的是将crypto_all.py项目改造成MLflow项目。

## 项目环境需求：

* MLflow

    https://mlflow.org/docs/latest/quickstart.html

* g++

## 改造过程：

1. [crypto_all.py](https://github.com/highwayns/FinRL/blob/master/finrl/meta/env_cryptocurrency_trading/crypto_all.py)在本项目的备份是 `origin/crypto_all.py`

2. 按照MLFlow的项目规范改造，得到 `crypto_all.py`

3. 运行项目

    1. 使用MLFlow项目方式运行 

            mlflow run <path_to_repo>/cryptocurrency_trading

    1. 使用Python文件方式运行 

            ipython <path_to_repo>/cryptocurrency_trading/main.py

## 改造结果

1. 支持MLFlow项目方式运行，无需手动安装依赖，使用MLflow来初始化conda环境，模型可以正常导出

2. 原Finrl项目的问题和对策：

      Q1. Finrl项目引用了ElegantRL项目，而ElegantRL项目在2023年1月8日做了版本更新。Finrl项目中的ElegantRL的agent代码并没有更新。

      A1. 修改了Finrl项目对ElegantRL的引用，在requirements和setup.py中ElegantRL项目指向和去年的 [superyuri/FinRL commit@2f1f65b](https://github.com/superyuri/FinRL/commit/2f1f65b58a10fc7193934e02de74112deee7056c)

      Q2. 由于未知的原因，当前代码在模型为A2C模式下，DRL库stablebaseline3运行会出错（错误信息应该是，反向传递的时候字典为空，导致读取字典的值出现空引用异常）。
      
      A2. 解决方案是魔改了stablebaseline3
[superyuri/stable-baselines3 commit@2b7e4f](https://github.com/superyuri/stable-baselines3/commit/2b7e4fdd52404300bead72089f26d1186f2e28ca)和[superyuri/stable-baselines3 commit@d28343e](https://github.com/superyuri/stable-baselines3/commit/d28343e2c237ca53033b38aa2efea85c7dcc6ae9)，并在Finrl一侧更新了引用
[superyuri/FinRL commit@c642c66](https://github.com/superyuri/FinRL/commit/c642c66606cfc23394a8d5cc2c9744c6f46ff964)

2. 在Mlflow改造过程中，可以将程序运行的过程中产生的数据导出到工件之中，其中`纯文本`，`DataFrame`和`Plot`是典型的输出对象，目前解决方案如下：

    1. `纯文本`: 本项目将`stdout`和`stderr`修改了输出位置，`finrl`库函数产生的文本输出到了`mlflow artifacts`路径。
    2. `DataFrame`: 可以考虑输出到csv文件中。目前有些代码是保存到当前目录，可以通过工具函数统一输出到`mlflow artifacts`路径。
    3. `Plot`: 可以考虑输出到图片文件中。目前有些代码是保存到当前目录，可以通过工具函数统一输出到`mlflow artifacts`路径。


#### 问题排查：

1. 如果mlflow环境初始化conda的时候pip包安装出错`CondaEnvException: Pip failed`，可以尝试清理pip

        pip cache purge

#### 实用工具

1. 清理conda，除了base环境外全部清除

        for i in `conda env list|awk '{print $1}'|egrep -v 'base|#'|tr '\n' ' '`;do echo $i;conda env remove --name $i;done

#### 存在的问题：


1. 程序输出的文件位置应该可以配置，由于引用的DataProcessor中'gmo'数据源的据直接保存到当前目录，会导致程序和数据混乱。

2. 在以下5中模型[a2c/ddpg/ppo/td3/sac]中,由于a2c修改过sb3的源码，程序不出错的模型和引擎为`multiple_a2c`，`advance_a2c`，`multiple_ppo`和`advance_ppo`。其他[ddpg/td3/sac]与[multiple/advance]的排列组合都会出错。错误信息是`AttributeError: 'NoneType' object has no attribute 'get'`.[dqn]模型的错误是`NotImplementedError: NotImplementedError`

3. multiple_a2c，advance_a2c模型的Plot输出中，显示的结果感觉不是正确。