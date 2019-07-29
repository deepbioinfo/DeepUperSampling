# DeepUperSampling
## File details
- SemisupCallback.py
A call back class is used in semisupervise learning
- SemisupController.py
A class is used to calculate the weight in each epoch
- semisupLearner_keras.py
To implement semi-supervise learning by keras
- nets.py
To define yourself network
- main.py
A example to apply these files
- conf.ini
A config file to config some parameters
## Usage
用户只需定义自己的网络结构并在main函数中使用。一个使用的例子如下。
### Step 1. 定义网络
如在nets.py中定义了一个网络函数semisup_net(),仿照在此函数定义自己的网络。
### Step 2. 在主函数中使用
如示例程序main.py的函数semisupLearn中，在65行，先获得一个网络，然后66-78行读入参数，79行把网络和参数传给SemisupLearner的构造函数，81行通过构造的对象训练及预测。
仿照此示例编程，您只需在nets.py中定义自己的semisup_net网络函数，再在main函数中load自己的数据即可。（参考95-112行）
## 配置文件说明
conf.ini中保存了网络训练需要的参数及文件路径的参数。网络训练参数有两个节点netparam和semisupparam，用户只能修改关键字的值，不能添加或删除；在给出的示例中有3个文件名称的节点，你可以删除或修改。
