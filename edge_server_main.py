import argparse
import cv2
import torchvision.transforms as transforms
import torch

from PIL import Image

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from communication import serverCommunication


WINDOW_NAME = 'CameraDemo'


def parse_args():
    # Parse input arguments
    desc = 'ANS in edge server side'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dnn', dest='dnn_model',
                        help='vgg, yolo',
                        default='yolo', type=str)
    parser.add_argument('--host', dest='host',
                        help='Ip address',
                        default='0.0.0.0', type=str)
    parser.add_argument('--port', dest='port',
                        help='Ip port',
                        default=8080, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
    else:
        model = tinyYolo()
        model.eval()

    model.to(device)

    communication = serverCommunication(args.host, args.port)

    while True:
        try:
            conn, addr = communication.accept_conn()
            with conn:
                recv_data = communication.receive_msg(conn)
                if recv_data is None:
                    continue

                # Check if client sent an end-of-experiment sync packet
                if isinstance(recv_data, dict) and recv_data.get('type') == 'SYNC_FILE':
                    filename = recv_data['filename']
                    content = recv_data['content']
                    import os
                    save_dir = os.path.join(os.path.dirname(__file__), 'experiments', 'data')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, filename)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"\n[EDGE SERVER] Successfully captured client experiment data: {filename} ({len(content)} bytes)")
                    print(f"[EDGE SERVER] Saved locally to: {save_path}\n")
                    communication.send_msg(conn, "ACK_SYNC")
                    continue

                partition_point = recv_data[0]
                data = recv_data[1]
                data = torch.autograd.Variable(data)
                prediction = model(data.to(device), server=True, partition=partition_point)
                res = prediction.data.cpu()

                msg = communication.send_msg(conn, res)

        except (KeyboardInterrupt, TypeError, OSError) as e:
            communication.close_channel()

