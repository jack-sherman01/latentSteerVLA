from setuptools import setup, find_packages

setup(
    name="compsteer",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "gymnasium>=0.29.0",
        "mani_skill>=3.0.0",
        "lerobot>=0.1.0",
        "transformers>=4.40.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.15.0",
        "wandb>=0.16.0",
        "h5py>=3.9.0",
        "imageio>=2.31.0",
        "opencv-python>=4.8.0",
    ],
    extras_require={
        "groot": [
            "groot @ git+https://github.com/NVIDIA/Isaac-GR00T.git",
        ],
        "rdt": [
            "rdt @ git+https://github.com/TeleHuman/RDT-1B.git",
        ],
    },
)
