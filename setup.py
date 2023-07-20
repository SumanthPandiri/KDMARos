from setuptools import setup

package_name = 'kdmapackage'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sumanth Pandiri',
    maintainer_email='pandiri05@gmail.com',
    description='KDMA in ROS package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kdma = kdmapackage.kdma:main'
        ],
    },
)
