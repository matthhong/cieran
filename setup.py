from setuptools import setup, find_packages

setup(
    name='Cieran',
    version='0.0.1',
    description='Designing Colormaps with a Teachable Robot',
    author='Matt-Heun Hong',
    packages=find_packages(),
    install_requires=[
        'coloraide==2.2.2',
        'ipywidgets>=8.0.4',
        'matplotlib>=3.7.0',
        'networkx>=2.8.8',
        'numpy>=1.24.2',
        'scipy>=1.10.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)