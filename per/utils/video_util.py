"""
Video util.
"""

import os
import uuid

import torch as tc
import numpy as np

import moviepy.editor as mpy


def _collect_footage(env, q_network, max_frames):
    frames = []
    with tc.no_grad():
        o_t = env.reset()
        for t in range(0, max_frames):
            ### act.
            a_t = q_network.sample(
                x=tc.FloatTensor(o_t).unsqueeze(0),
                epsilon=0.01)
            a_t = int(a_t.squeeze(0).detach().numpy())
            o_tp1, r_t, done_t, _ = env.step(action=a_t)
            if done_t:
                o_tp1 = env.reset()
            frames.append(np.array(env.render(mode='rgb_array')))
            o_t = o_tp1
    return frames


def _make_video(frames, base_path, fps, max_frames):
    def make_frame(t):
        # t will range from 0 to (self.max_frames / self.fps).
        frac_done = t / (max_frames // fps)
        max_idx = len(frames) - 1
        idx = int(max_idx * frac_done)
        x = frames[idx]
        return frames[idx]

    filename = f"{uuid.uuid4()}.gif"
    fp = os.path.join(base_path, filename)

    clip = mpy.VideoClip(make_frame, duration=(max_frames // fps))
    clip.write_gif(fp, fps=fps)
    return fp


def save_video(env, q_network, checkpoint_dir, run_name):
    base_path = os.path.join(checkpoint_dir, run_name, 'videos')
    os.makedirs(base_path, exist_ok=True)
    frames = _collect_footage(
        env=env,
        q_network=q_network,
        max_frames=2048)
    fp = _make_video(
        frames=frames,
        base_path=base_path,
        fps=64,
        max_frames=2048)
    print(f"Saved video to {fp}")
