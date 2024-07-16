from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="classification-model",
    version="0.1.0",
    author="Andrew Marous",
    author_email="andrew.marous@gmail.com",
    description="A local server that accepts an image file of a manufactured product and returns a prediction vector "
                "of the product's SKU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrewmarous/opsmart-product-classifier",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "torch",
        "torchvision",
        "numpy==1.26.4",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'model-server=server:main',
        ],
    },
)