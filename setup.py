from setuptools import setup, find_packages

with open('README.md', 'r+', encoding="utf8") as f:
    description = f.read()

setup(
    name='hcr_nn',
    version='1.0.4',
    description='Package implementing Hierarchical Correlation Reconstruction Methodologies',
    packages=find_packages(),
    #setup_requires=['pytest-runner', 'flake8'],
    #tests_require=['pytest'],
    install_requires=[
        'numpy>=1.26.4',
        'matplotlib>=3.10.1',
        'scipy>=1.13.1',
        'torch>=2.7.0',
        'pandas>=2.3.0',
    ],
    entry_points={
        "console_scripts": [
            "hcr_nn_info = hcr_nn:layers.hcr_nn_info",
        ],
    },
    long_description=description,
    long_description_content_type='text/markdown',
)