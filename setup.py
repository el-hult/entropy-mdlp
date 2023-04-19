from setuptools import setup, find_packages

setup(
    name="entropymdlp",
    version="0.1",
    description="Minimum description length principle",
    author="Ludvig Hult",
    author_email="ludvig.hult@gmail.com",
    install_requires=["numpy", "numba"],
    packages=find_packages(),
    license="MIT",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
