from setuptools import setup


setup(
    name="pytorch_per",
    py_modules=["per"],
    version="0.3.0",
    description="A Pytorch implementation of Prioritized Experience Replay.",
    author="Lucas D. Lingle",
    install_requires=[
        'torch==1.10.0',
        'numpy==1.20.3',
        'mpi4py==3.0.3',
        'moviepy==1.0.3',
        'atari-py==0.2.6',
        'gym==0.18.0',
        'opencv-python==4.5.2.52',
    ]
)
