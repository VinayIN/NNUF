from setuptools import find_packages, setup

package_name = "seld_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "librosa",
        "opencv-python",
        "Pillow",
        "torch",
        "torchvision",
        "tqdm",
        "tensorboard",
    ],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="Package description",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "seld = seld_pkg.inference:main",
        ],
    },
)
