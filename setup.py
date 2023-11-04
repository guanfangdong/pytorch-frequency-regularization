from setuptools import setup, find_packages

setup(
    name='frequency_regularization',
    version='0.1.0',
    description='Frequency Regularization Python package',
    url='https://github.com/guanfangdong/pytorch-frequency-regularization',
    author='Zhao, Chenqiu and Dong, Guanfang and Zhang, Shupei and Tan, Zijie and Basu, Anup',
    # license='BSD 2-clause',
    packages=find_packages(where="src"),
    install_requires=['numpy',
                      'imageio',
                      'torch',
                      'matplotlib',
                      ],
    python_requires=">=3.7, <4",







    classifiers=[
    #     'Development Status :: 1 - Planning',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',
    #     'Operating System :: POSIX :: Linux',
    #     'Programming Language :: Python :: 2',
    #     'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.4',
    #     'Programming Language :: Python :: 3.5',
    ],
)
