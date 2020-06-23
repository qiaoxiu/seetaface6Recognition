# seetaface6Recognition
基于中科院最新开源的模型提供python实现的人脸全流程方案 人脸定位 人脸识别 人脸登录
SeetaFace6包含人脸识别的基本能力：人脸检测、关键点定位、人脸识别，同时增加了活体检测、质量评估、年龄性别估计，并且顺应实际应用需求，开放口罩检测以及口罩佩戴场景下的人脸识别模型。


开放标准C++开发接口，涵盖x86和ARM架构支持，未来会逐步开放Ubuntu、CentOS、macOS、Android、IOS的支持。

开源地址：https://github.com/seetafaceengine/SeetaFace6





当前版本是根据开源的python版本进行实现 人脸全流程 提供的应用，识别效果高于facenet模型， 对同一个人图片相似度高  不同人相似度极低  复合要求
so文件  链接：https://pan.baidu.com/s/1cA0UbI1wUmcI0fhJHVyLYQ 
提取码：ifoo

model文件 链接：https://pan.baidu.com/s/1Nh5jt1P-kxxLzjYP7B-VMw 
提取码：psfm


将这俩个文件解压放到seetaface/lib/ 目录下



人脸识别流程： 1.首先通过接口进行人脸注册，进行人脸Embedding 然后人脸库数据存储 2.人脸识别 通过Embedding 在人脸库进行搜索匹配最佳的人脸图像 给前端显示

首先初始化一部分人脸进行注册 提供的都是明显的人脸图片 python test4.py

然后启动app文件 启动服务 就可以进行页面 和接口 的人脸注册 和人脸识别
