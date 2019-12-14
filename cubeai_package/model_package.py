# -*- coding: utf-8 -*-
from ucumos.session import AcumosSession
from ucumos.modeling import Model, List
from ucumos.metadata import Requirements
from model_service import ModelService

# 创建模型服务对象
model_service = ModelService()

# 封装模型API接口函数： 初始化模型数据。该接口函数仅供系统调用，必须存在，不要修改或删除！！！
def init_model_data(text: str) -> str:
    model_service.init_model_data()
    return 'model_data_loaded'

# 封装模型API接口函数
def classify(img_list: List[float]) -> List[float]:
    """
    手写数字识别，输入：归一化之后灰度值数组
    """
    return model_service.classify(img_list)


# 打包模型文件
model = Model(init_model_data=init_model_data,
              classify=classify)
session = AcumosSession()
reqs = Requirements(reqs=['tensorflow==1.12.2'],
                    scripts=['./model_service.py'],
                    packages=['./core'])
session.dump(model, 'mnist示例', './out/', reqs)
