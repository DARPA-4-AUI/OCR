## 本项目基于[yolo3](https://github.com/pjreddie/darknet.git) 与[crnn](https://github.com/meijieru/crnn.pytorch.git)  实现中文自然场景文字检测及识别  

## 环境部署

GPU部署 参考:setup.md     
CPU部署 参考:setup-cpu.md     

## 模型选择  
``` Bash
参考config.py文件
```  

## 构建docker镜像 
``` Bash
##下载Anaconda3 python 环境安装包（https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh） 放置在chineseocr目录下   
##建立镜像   
docker build -t chineseocr .   
##启动服务   
docker run -d -p 8080:8080 chineseocr /root/anaconda3/bin/python app.py

```

## web服务启动
``` 
python app.py
```

## 访问服务
## 要用的话要改一下里边图片的路径，或者改成传进去路径[file_path]
```
python client.py
```

## 结果
结果（json格式文件）会存在result文件夹下，名字跟图片相同

## 结果示例
在result文件夹底下有一对结果示例

