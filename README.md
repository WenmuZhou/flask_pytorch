# flask_pytorch
using flask to run pytorch model

# start server
```sh
python3 app.py
```
you will see a result like this

![flask](img/flask.jpg)
# Submitting requests to pytorch server
```sh
python3 request_sample.py -f='file_path'
```
send a image like this

![send_image](img/1.jpg)

you will see a result like this (I use the mnist image as example).

![image](img/result.jpg)

the demo model can de download in [baiduyun](https://pan.baidu.com/s/1Y5zmNoo9ZGTfLmx5Plr83A)

# demo online

After starting the server, use http://127.0.0.1:5000/demo for online testing

![online testing](img/online_test.jpg)

# Acknowledgement
This repository refers to [deploy-pytorch-model](https://github.com/L1aoXingyu/deploy-pytorch-model), and thank the author again.