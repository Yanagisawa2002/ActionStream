from glob import glob
from setuptools import find_packages, setup


PACKAGE_NAME = "action_stream_policy"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.json")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ActionStream maintainers",
    maintainer_email="cgliu@localhost",
    description="Deterministic 7D reach-and-lift policy adapter for ActionStream M7.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "scripted_policy_node = action_stream_policy.policy_node:main",
        ],
    },
)
