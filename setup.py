from setuptools import setup, find_packages

setup(
    name='frereg',
    version='0.1.0',
    description='Frequency Regularization Python Package',
    url='https://github.com/guanfangdong/pytorch-frequency-regularization',
    author='Zhao, Chenqiu and Dong, Guanfang and Zhang, Shupei and Tan, Zijie and Basu, Anup',
    packages=['frereg'],
    license="Apache License 2.0",
    install_requires=['numpy',
                      'imageio',
                      'torch',
                      'matplotlib',
                      ],
    python_requires=">=3.8, <3.12",

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
