# coding:utf-8

import socket
import json
import base64
import numpy as np
from PIL import Image
from multiprocessing import Process
from keras import backend as K
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras.models import load_model
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections


class HTTPServer(object):
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        self.server_socket.listen(20)
        print("start listen ...")
        while True:
            client_req, client_address = self.server_socket.accept()
            print("[%s, %s]用户连接上了" % client_address)
            handle_client_process = Process(target=handle_client,
                                            args=(client_req, ))
            handle_client_process.start()
            # client_req.close()

    def bind(self, port):
        self.server_socket.bind(("", port))


def handle_client(client_req):
    """
    处理客户端请求
    """
    # 获取客户端请求数据
    req_buff = client_req.makefile('rb')
    # 解析请求报文
    method, route, params, http_version = parse_first_line(req_buff)
    # print("method: " + method)
    # print("route: " + route)
    # print("params: " + str(params))
    # print("http_version: " + http_version)
    headers = parse_headers(req_buff)
    # print('headers: ' + str(headers))
    data = parse_body(req_buff, headers)
    # print('data: ' + data)
    body_content = json.loads(data)
    images_numpy = []
    if 'path' in body_content:
        image_path = body_content['path']
        print("load file from local: " + image_path)
        with Image.open(image_path) as image:
            images_numpy.append(np.array(image, dtype=np.uint8))
    elif 'image' in body_content:
        image_content = body_content['image']
        print("load file from request body.")
        image_asc = image_content.endoce('ascii')
        image_decode = base64.b64decode(image_asc)
        images_numpy.append(np.frombuffer(image_decode, dtype=np.uint8))
    images_numpy = np.array(images_numpy)
    y_pred = model.predict(images_numpy)
    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.005,
                                       top_k=200,
                                       normalize_coords=True,
                                       img_height=img_height,
                                       img_width=img_width)
    pred_labels = set(y_pred_decoded[0][..., 0].astype(np.int8).astype(np.str).tolist())
    # label_names = [classes[int(lb)] for lb in pred_labels]
    # 构造响应数据
    response_body = ','.join(pred_labels)
    response_start_line = "HTTP/1.1 200 OK\r\n"
    response_headers = "Server: SDZW_SSD\r\nContent-Length:" + str(len(response_body)) + "\r\n"
    response = response_start_line + response_headers + "\r\n" + response_body
    print("response data:", response)

    # 向客户端返回响应数据
    client_req.send(bytes(response, "utf-8"))

    # 关闭客户端连接
    client_req.close()


def parse_first_line(req_buff):
    request_line = req_buff.readline()
    method, route, http_version = request_line.decode().strip().split()
    route, params = parse_param(route)
    return method, route, params, http_version


def parse_headers(req_buff):
    headers = {}
    while True:
        header = req_buff.readline().decode()
        head = header.split(':', 1)
        if len(head) == 1:
            break
        k, v = head
        headers[k.strip()] = v.strip()
    return headers


def parse_body(req_buff, headers):
    if 'Content-Length' not in headers:
        return '{}'
    content_length = int(headers['Content-Length'])
    data = req_buff.read(content_length).decode()
    return data


def parse_param(route):
    params = {}
    routes = route.split('?')
    if len(routes) != 1:
        param_info = routes[1].split('&')
        for param in param_info:
            k, v = param.split('=')
            params[k.strip()] = v.strip()
    return routes[0], params


def main():
    http_server = HTTPServer()
    http_server.bind(8135)
    http_server.start()


img_height = 312  # 图像的高度
img_width = 416  # 图像的宽度
# 设置要加载的模型的路径.
model_path = '../ssd7_v3_epoch-38_loss-0.9415_val_loss-0.6735.h5'
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
K.clear_session()  # 从内存中清理曾经加载的模型.
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})

if __name__ == "__main__":
    main()
