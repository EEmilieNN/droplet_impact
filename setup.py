from setuptools import setup, find_packages

setup(
    name='droplets_impact_speeds',
    version='0.1.0',
    packages=find_packages(where="model_python"),
    install_requires=[], 
    package_dir={'': 'model_python'},
    author='Emilien Gouffault',
    author_email='emilien.gouffault@gmail.com',
    description='This package was created for a DTU Wind&Energy Systems project to calculate the impact speeds of droplets on wind turbine blades.',
    url='https://github.com/EEmilieNN/droplets_impact_speeds',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
