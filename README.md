# CSS代码解读

原来的代码连接：https://github.com/yanxinzju/CSS-VQA </br>

首先，大致解读一下每个py文件的内容。本文重点与创新点CSS体现在train.py

attention.py </br>
  具有两个方法：attention()方法是直接拼接，newattention（）方法是q,v对应相乘</br>
base_model.py </br>
  使用基线模型进行训练，（使用了上面的attention和classifier） </br>
classifier.py </br>
  简单的分类 </br>
dataset.py </br>
   加载相关数据集的问题与答案，并处理成对应的元组 </br>
    获取并处理VQA特征数据集
eval.py
  验证
fc.py
  全连接
language_model.py
  处理词嵌入或问题嵌入
main.py
  主函数，这个主要是来自另一篇论文，使用了LMH偏差计算方法
train.py
  这里是本文的重点与创新点，CSS的方法再此体现
util.py
  一些工具函数
vqa_debias_loss_function.py
  处理偏差的函数
