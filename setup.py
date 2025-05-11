from setuptools import setup, find_packages

setup(
    name="advances_project",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "lmdb",
        "matplotlib",
    ],
)
