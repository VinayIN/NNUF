from setuptools import find_packages, setup


package_name = 'talker_pkg'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='binay',
    maintainer_email='contact@binaypradhan.com',
    description='Publisher: talker package.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'talker = talker_pkg.talker:main',
        ],
    },
)