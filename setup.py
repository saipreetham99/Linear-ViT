from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="linear-vit",
    version="0.1.0",
    author="Your Name",
    description="Sub-quadratic Vision Transformers with RippleAttention and HydraAttention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "timm>=0.9.0",
        "einops>=0.6.1",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
    ],
)
