# flask_pytorch
using flask to run pytorch model

# start server
```sh
python3 app.py
```
you will see a result like this

![flask](static/img/flask.jpg)
# Submitting requests to pytorch server
```sh
python3 request_sample.py -f='file_path'
```
send a image like this

![send_image](static/img/1.jpg)

you will see a result like this (I use the mnist image as example).

![image](static/img/result.jpg)

the demo model can de download in [baiduyun]https://pan.baidu.com/s/16O5eFaGoB9VZA7goBTNO4w)

# demo online

After starting the server, use http://127.0.0.1:5000/demo for online testing

![online testing](static/img/online_test.jpg)

# Acknowledgement
This repository refers to [deploy-pytorch-model](https://github.com/L1aoXingyu/deploy-pytorch-model), and thank the author again.