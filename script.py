"""
Script to operate DQN agents trained with Prioritized Experience Replay.
"""

import argparse


def create_argparser():
    parser = argparse.ArgumentParser(
        description="Pytorch implementation of Prioritized Experience Replay")
    parser.add_argument("--mode", choices=["train", "evaluate", "video"])
    parser.add_argument("--double_dqn", choices=[0,1], default=1)
    parser.add_argument("--dueling_head", choices=[0,1], default=0)
    return parser
