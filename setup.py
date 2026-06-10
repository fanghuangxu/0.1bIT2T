from setuptools import setup, find_packages

setup(
    name='mini-dialog-ai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.30.0',
        'pillow>=10.0.0',
        'numpy>=1.24.0',
        'accelerate>=0.20.0',
        'sentencepiece>=0.1.99',
    ],
    entry_points={
        'console_scripts': [
            'mini-ai=mini_dialog_ai.cli:main',
        ],
    },
)