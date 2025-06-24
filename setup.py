from setuptools import setup, find_packages

setup(
    name='droplet_impact',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib','scikit-learn','joblib','pandas','time'],
    package_data={
        'droplet_impact': ['data/*.pkl'],
    },
    author='Emilien Gouffault',
    author_email='emilien.gouffault@gmail.com',
    description='This package was created for a DTU Wind&Energy Systems project to calculate the impact speeds of droplets on wind turbine blades.',
    url='https://github.com/EEmilieNN/droplet_impact',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
