from setuptools import setup

package_name = 'arm_circle_publisher'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@todo.todo',
    description='Publishes a circular joint trajectory for a robot arm.',
    license='Apache-2.0',
    tests_require=['pytest'],
   entry_points={
    'console_scripts': [
        'circle_publisher = arm_circle_publisher.circle_publisher:main',
    ],
    },
)