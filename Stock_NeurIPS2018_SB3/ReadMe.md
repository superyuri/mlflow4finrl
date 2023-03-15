# Stock_NeurIPS2018_SB3_mlflow

## 简介

本项目是由[Finrl例程：Stock_NeurIPS2018_SB3.ipynb](https://github.com/AI4Finance-Foundation/FinRL/blob/master/examples/)的Notebook项目修改而来。
目的是将Notebook项目改造成MLflow项目。

## 项目环境需求：

* MLflow

    https://mlflow.org/docs/latest/quickstart.html

* g++
* nbconvert (notebook格式转化为Python)

## 改造过程：

1. [Finrl例程：Stock_NeurIPS2018_SB3.ipynb](https://github.com/AI4Finance-Foundation/FinRL/blob/master/examples/)在本项目的备份是 `nokebook/Stock_NeurIPS2018_SB3.ipynb`

2. 使用nbconvert将notebook格式转化为Python文件

        jupyter nbconvert --to python <path_to_repo>/Stock_NeurIPS2018_SB3/notebook/Stock_NeurIPS2018_SB3.ipynb --execute

    生成后的文件是 `nokebook/Stock_NeurIPS2018_SB3.py`

3. 测试生成的文件，由于使用了notebook的一些特性。需要使用ipython命令启动程序

        ipython <path_to_repo>/Stock_NeurIPS2018_SB3/nokebook/Stock_NeurIPS2018_SB3.py

4. 按照MLFlow的项目规范改造，得到 `Stock_NeurIPS2018_SB3_mlflow.py`

5. 运行项目

    1. 使用MLFlow项目方式运行 

            mlflow run <path_to_repo>/Stock_NeurIPS2018_SB3

    1. 使用Python文件方式运行 

            ipython <path_to_repo>/Stock_NeurIPS2018_SB3/Stock_NeurIPS2018_SB3_mlflow.py

## 改造结果

1. 支持MLFlow项目方式运行，无需手动安装依赖，使用MLflow来初始化conda环境，模型可以正常导出

2. 由于原项目是Notebook文件，结果输出的类型有`纯文本`，`DataFrame`和`Plot图片`，在改造成MLflow项目时会有一定的问题。这些结果输出依赖于`finrl`库函数，对`finrl`库函数的修改或者使用替代函数需要一定的二次开发。所以目前解决方案如下：

    1. `纯文本`: 将`stdout`和`stderr`修改了输出位置，`finrl`库函数产生的文本输出到了`mlflow artifacts`路径。
    2. `DataFrame`: 对于源码文件中的`DataFrame`，目前的替代方案是输出到csv文件中，对于`finrl`库函数产生的`DataFrame`输出是`<IPython.core.display.HTML object>`，目前没有很好的解决方法。
    3. `Plot图片`: 源码中没有`Plot图片`输出，`finrl`库`backtest_plot()`产生的`Plot图片`输出是`<IPython.core.display.HTML object>`，目前没有很好的解决方法。


#### 问题排查：

1. 如果mlflow环境初始化conda的时候pip包安装出错`CondaEnvException: Pip failed`，可以尝试清理pip

        pip cache purge

#### 实用工具

1. 清理conda，除了base环境外全部清除

        for i in `conda env list|awk '{print $1}'|egrep -v 'base|#'|tr '\n' ' '`;do echo $i;conda env remove --name $i;done
#### 下一步工作

1. 模型应该支持 a2c/ddpg/ppo/td3/sac 五种模式，目前代码支持sac模式，其他模式需要简单改动源码，未来通过输入参数指定模式。
2. 异常处理的开发

#### 存在的问题：

1. 原项目是Notebook，`DataFrame`和`Plot图片`的输出方式改造。


    
