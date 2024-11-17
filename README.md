## The project is based on [yolo3](https://github.com/pjreddie/darknet.git) and [crnn](https://github.com/meijieru/crnn.pytorch.git) for Chinese natural scene text detection and recognition. 

## Environment Setup

For GPU deployment, refer to: setup.md
For CPU deployment, refer to: setup-cpu.md 

## Model Selection  

``` Bash
Refer to the config.py file
```

## Building Docker Image

``` Bash
## Download Anaconda3 Python environment installation package (https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh) and place it in the chineseocr directory   
## Build the image   
docker build -t chineseocr .   
## Start the service   
docker run -d -p 8080:8080 chineseocr /root/anaconda3/bin/python app.py
```

## Starting the Web Service

``` 
python app.py
```

## Accessing the Service

## To use this, you need to modify the image paths inside, or change it to accept image paths as an argument [file_path]

```
python client.py
```

## Result

The results (in json format) will be stored in the result folder with the same name as their corresponding images.

## Result Example

There is an example result in the result folder.
