from setuptools import setup, find_packages


def get_long_description():
    with open('README.md') as f:
        return f.read()


setup(
    name='movingclusterlib',
    version='0.1.2',
    author='Alexander Khlebushchev',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    install_requires=[
        'matplotlib>=3.6.0',
        'numpy>=1.23.3',
        'pandas>=1.5.0',
        'datatool-python>=1.3.4',
    ],
)
