from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "action_stream_isaac"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ActionStream maintainers",
    maintainer_email="cgliu@localhost",
    description="Headless Isaac Sim Franka adapters for ActionStream M7 and M8.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "capability_probe = action_stream_isaac.capability_probe:main",
            "isaac_adapter = action_stream_isaac.isaac_adapter:main",
            "dynamic_isaac_adapter = action_stream_isaac.dynamic_isaac_adapter:main",
        ],
    },
)
