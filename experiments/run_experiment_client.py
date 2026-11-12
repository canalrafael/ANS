"""
ANS Adversarial Split Inference Experiment Runner (1000 Frames)
===============================================================
Executes automated 1000-frame stress/attack experiments on the Raspberry Pi
across 3 phase scenarios and 3 attack types:

Scenarios:
  - scenario_a: 50% Phase 1 (1..500), 25% Phase 2 (501..750), 25% Phase 3 (751..1000)
  - scenario_b: 33% Phase 1 (1..330), 33% Phase 2 (331..660), 34% Phase 3 (661..1000)
  - scenario_c: 65% Phase 1 (1..650), 10% Phase 2 (651..750), 25% Phase 3 (751..1000)

Attacks:
  - memory:  Microarchitectural cache & memory bus contention (IAES stressor)
  - cpu:     Multi-core CPU compute saturation (100% load on Cortex-A72)
  - network: Injected network latency/jitter degradation

Features:
  - Automatic phase transitions and attack injection.
  - Real-time local CSV data logging on the Raspberry Pi.
  - Automatic end-of-run synchronization: copies full CSV to PC edge server!
"""

import argparse
import json
import os
import pickle
import sys
import time
import cv2
import numpy as np
import torch

# Ensure parent directory is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from client_camera_main import vgg_info, yolo_info
from keyFrameDetection import KeyFrameDetection
from muLinUCB import muLinUCB
from communication import clientCommunication

from experiments.stressor import create_stressor, NetworkDelayStressor
from experiments.data_logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description="ANS 1000-Frame Stress & Adaptation Experiment")
    parser.add_argument('--scenario', type=str, required=True,
                        choices=['scenario_a', 'scenario_b', 'scenario_c'],
                        help="scenario_a (50/25/25), scenario_b (33/33/34), scenario_c (65/10/25)")
    parser.add_argument('--attack', type=str, required=True,
                        choices=['memory', 'cpu', 'network'],
                        help="Attack type: memory, cpu, or network")
    parser.add_argument('--total-frames', type=int, default=1000,
                        help="Total number of frames (default: 1000)")
    parser.add_argument('--dnn', dest='dnn_model', type=str, default='vgg',
                        choices=['vgg', 'yolo'], help="DNN model (vgg or yolo)")
    parser.add_argument('--host', type=str, default='192.168.1.19',
                        help="Edge server IP address")
    parser.add_argument('--port', type=int, default=8888,
                        help="Edge server port")
    parser.add_argument('--video-dev', type=int, default=0,
                        help="V4L2 camera device index (default: 0)")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of contention workers (default: 4 for RPi4)")
    parser.add_argument('--no-display', action='store_true', default=True,
                        help="Run in headless mode without X11 GUI")
    return parser.parse_args()


def get_scenario_bounds(scenario: str, total_frames: int):
    """Calculates phase transition boundaries."""
    if scenario == 'scenario_a':  # 50% / 25% / 25%
        p1_end = int(total_frames * 0.50)
        p2_end = int(total_frames * 0.75)
    elif scenario == 'scenario_b':  # 33% / 33% / 34%
        p1_end = int(total_frames * 0.33)
        p2_end = int(total_frames * 0.66)
    elif scenario == 'scenario_c':  # 65% / 10% / 25%
        p1_end = int(total_frames * 0.65)
        p2_end = int(total_frames * 0.75)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return p1_end, p2_end


def get_phase_info(frame: int, p1_end: int, p2_end: int):
    """Returns phase name and whether attack should be active."""
    if frame <= p1_end:
        return 'Phase 1 - Nominal', False
    elif frame <= p2_end:
        return 'Phase 2 - Attack', True
    else:
        return 'Phase 3 - Recovery', False


def prepare_image_vgg(frame):
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image = image.transpose(2, 0, 1)
    img = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    return img


def prepare_image_yolo(frame):
    image = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    img = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return img


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def sync_to_server(host: str, port: int, filename: str, content: str):
    """Sends synchronization packet to edge server to save CSV directly on the PC."""
    print(f"\n[SYNC] Uploading {filename} to Edge Server ({host}:{port})...")
    try:
        comm = clientCommunication(host, port)
        sync_packet = {
            'type': 'SYNC_FILE',
            'filename': filename,
            'content': content
        }
        comm.send_msg(sync_packet)
        ack = comm.receive_msg()
        comm.close_channel()
        print(f"[SYNC] Successfully synchronized to PC! Server reply: {ack}")
        return True
    except Exception as e:
        print(f"[SYNC WARNING] Could not auto-sync to PC server ({e}). Data remains safely saved locally on Raspberry Pi.")
        return False


def main():
    args = parse_args()
    p1_end, p2_end = get_scenario_bounds(args.scenario, args.total_frames)

    print("=" * 80)
    print("ANS ADVERSARIAL SPLIT INFERENCE EXPERIMENT")
    print(f"Scenario:     {args.scenario} (P1: 1..{p1_end} | P2: {p1_end+1}..{p2_end} | P3: {p2_end+1}..{args.total_frames})")
    print(f"Attack Type:  {args.attack.upper()}")
    print(f"Total Frames: {args.total_frames}")
    print(f"Model:        {args.dnn_model.upper()}")
    print(f"Server:       {args.host}:{args.port}")
    print("=" * 80)

    # Initialize model
    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
        frontEndDelay = load_obj('models/vgg16FrontEndDelay')
        partitionInfo = vgg_info
    else:
        model = tinyYolo()
        model.eval()
        frontEndDelay = load_obj('models/yoloFrontEndDelay')
        partitionInfo = yolo_info

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    action_num = len(partitionInfo)

    mu_controller = muLinUCB(mu=0.25, layerInfo=partitionInfo, frontDelay=frontEndDelay)
    communication = clientCommunication(args.host, args.port)
    stressor = create_stressor(args.attack, num_workers=args.workers)

    # Initialize Data Logger
    exp_name = f"ans_{args.attack}_{args.scenario}"
    logger = ExperimentLogger(experiment_name=exp_name, total_frames=args.total_frames)
    print(f"[LOGGER] Local CSV output: {logger.filepath}")

    # Open Camera
    cap = cv2.VideoCapture(args.video_dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera on /dev/video{args.video_dev}!")
        sys.exit(1)

    key_detector = KeyFrameDetection(threshold=0.8)
    old_frame = None
    current_doubling_frame = 0

    attack_active_last = False

    try:
        for frame_idx in range(1, args.total_frames + 1):
            phase_name, should_attack = get_phase_info(frame_idx, p1_end, p2_end)

            # Manage stressor state transitions
            if should_attack and not attack_active_last:
                print(f"\n>>> [TRANSITION] Entering {phase_name} at frame {frame_idx}! Activating {args.attack.upper()} stressor... <<<")
                stressor.start()
                attack_active_last = True
            elif not should_attack and attack_active_last:
                print(f"\n>>> [TRANSITION] Entering {phase_name} at frame {frame_idx}! Deactivating {args.attack.upper()} stressor... <<<")
                stressor.stop()
                attack_active_last = False

            # Capture camera frame
            ret, img = cap.read()
            if not ret or img is None:
                time.sleep(0.05)
                continue

            if args.dnn_model == 'vgg':
                preprocessed = prepare_image_vgg(img)
            else:
                preprocessed = prepare_image_yolo(img)

            # Doubling trick & forced sampling
            current_doubling_frame += 1
            if mu_controller.updateDoublingTrickFrameNum(current_doubling_frame):
                current_doubling_frame = 0

            # Keyframe check
            if frame_idx == 1 or old_frame is None:
                is_key = False
                old_frame = np.copy(img)
            else:
                is_key = key_detector.compare_images(old_frame, img)
                old_frame = np.copy(img)

            # Controller decision
            partition = mu_controller.getEstimationAction(is_key, current_doubling_frame)

            # 1. Local prefix computation
            t_local_start = time.time()
            if partition == action_num - 1:
                # Full local
                pred = model(preprocessed.to(device))
                t_local_end = time.time()
                local_delay = t_local_end - t_local_start
                offload_delay = 0.0
            else:
                intermediate = model(preprocessed.to(device), server=False, partition=partition)
                t_local_end = time.time()
                local_delay = t_local_end - t_local_start

                # Prepare tensor for transmission
                data_to_server = [partition, intermediate.data.cpu()]
                del intermediate

                # 2. Network offload
                t_offload_start = time.time()
                if isinstance(stressor, NetworkDelayStressor):
                    stressor.apply_delay()

                communication.send_msg(data_to_server)
                res = communication.receive_msg()
                communication.close_channel()
                t_offload_end = time.time()
                offload_delay = t_offload_end - t_offload_start

            # Update controller state (A and b)
            mu_controller.updateA_b(partition, offload_delay)

            # Log frame metrics
            record = logger.log_frame(
                frame=frame_idx,
                scenario=args.scenario,
                phase=phase_name,
                attack_type=args.attack if should_attack else 'none',
                partition=partition,
                offload_delay_s=offload_delay,
                local_delay_s=local_delay,
                is_key_frame=is_key,
                is_forced_sampling=False
            )

            # Print concise progress every frame
            print(f"[{frame_idx:04d}/{args.total_frames}] {phase_name:<19} | Part: {partition:2d} | Offload: {offload_delay:6.3f}s | Local: {local_delay:6.3f}s | E2E: {record['end_to_end_s']:6.3f}s | Avg: {record['avg_end_to_end_s']:6.3f}s")

    except KeyboardInterrupt:
        print("\n[STOP] Experiment interrupted by user.")
    finally:
        stressor.stop()
        cap.release()
        csv_text = logger.close()
        print(f"\n[DONE] Saved {len(logger.records)} frame records to local CSV: {logger.filepath}")

        # Sync to PC Edge Server automatically
        sync_to_server(args.host, args.port, logger.filename, csv_text)


if __name__ == '__main__':
    main()
