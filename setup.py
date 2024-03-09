from setuptools import setup, find_packages

install_requires = [
    'coloraide==1.4',
    'ipywidgets>=8.0.0',
    'matplotlib>=3.5.0',
    'networkx>=2.7.0',
    'numpy>=1.21.0',
    'scipy>=1.10.0'
]

dev_requires = [
    'altair',
    'jupyter',
    'vega-datasets',
    'mkdocs',
    'mkdocstrings[python]',
    'mkdocs-material'
]

setup(
    name='cieran',
    version='0.0.1',
    description='Designing Colormaps with a Teachable Robot',
    author='Matt-Heun Hong',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)