import json

import requests

file_path='./result/1.png'

img1=open(file_path,'rb')
file_resq={
    "file":img1
}
res=requests.post("http://127.0.0.1:8080/ocr",files=file_resq)
print(res.text)