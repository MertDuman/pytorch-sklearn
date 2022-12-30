from setuptools import find_packages, setup

setup(
    name="pytorch_sklearn",
    packages=find_packages(),
    version="0.1.0",
    description="Implements fit, predict and other useful ways of training with PyTorch.",
    author="Mert Duman",
    license="MIT",
    install_requires=["torch", "numpy", "matplotlib"]
)
