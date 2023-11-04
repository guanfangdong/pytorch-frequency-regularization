from setuptools import setup, find_packages

setup(
    name='freqreg',
    version='0.1.0',
    description='Frequency Regularization Python Package',
    url='https://github.com/guanfangdong/pytorch-frequency-regularization',
    author='Zhao, Chenqiu and Dong, Guanfang and Zhang, Shupei and Tan, Zijie and Basu, Anup',
    packages=['freqreg'],
    license="Apache License 2.0",
    install_requires=['numpy',
                      'imageio',
                      'torch',
                      'matplotlib',
                      ],
    python_requires=">=3.7, <4",
    # pytorch_requires=">=1.10.0, <2.1.0"

    classifiers=[
    #     'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
