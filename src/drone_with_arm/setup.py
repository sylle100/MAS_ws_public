from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drone_with_arm'


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        if not filenames:
            continue
        install_path = os.path.join('share', package_name, path)
        file_list = [os.path.join(path, filename) for filename in filenames]
        paths.append((install_path, file_list))
    return paths

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ] + package_files('models') + package_files('worlds'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kristoffer Jensen',
    maintainer_email='krik@mmmi.sdu.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
