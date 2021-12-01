from setuptools import setup


setup(
    name="pytorch_per",
    py_modules=["per"],
    version="0.0.1",
    description="A Pytorch implementation of Prioritized Experience Replay.",
    author="Lucas D. Lingle",
    install_requires=[
        'torch==1.8.1'
    ]
)