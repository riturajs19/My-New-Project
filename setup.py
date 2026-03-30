from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]:

    requirements: List[str] = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.1',
    packages=find_packages(),
    author = 'Rituraj',
    author_email = 'sriturajsingh745@gmail.com',
    install_requires= get_requirements('requirements.txt')
)