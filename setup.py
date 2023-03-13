from setuptools import setup

setup(
    name='tensorizer',
    version='0.0.1',
    description='A tool for PyTorch Module, Model, and Tensor Serialization/Deserialization.',
    url='https://github.com/coreweave/tensorizer',
    author='CoreWeave',
    license='MIT',
    packages=['tensorizer'],
    install_requires=[
        'torch>=1.9.0',
        'protobuf>=3.19.5',
    ],
    extras_require={
        'examples': [
            'transformers @ git+ssh://git@github.com/huggingface/transformers@d4ba6e1a0e8f662f3deadba25d982c6fb5fb772c',
            'diffusers==0.11.1'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
