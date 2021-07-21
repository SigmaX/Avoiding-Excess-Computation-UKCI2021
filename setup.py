from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='async_sim',
    version='0.1.0',
    packages=find_packages(),
    license='Academic',
    author='Eric O. Scott',
    author_email='ericsiggyscott@gmail.com',
    description='Experiments with asynchronous evolutionary algorithms.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    #url='https://github.com/SigmaX/async_2021',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Academic Free License (AFL)',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.6',
    install_requires=[
        'leap_ec==0.6',         # Evolutionary algorithms framework; need development version to be installed manually
    ],
    # entry_points={
    #     'console_scripts': ['evolve=bots_transfer.evolve:main',
    #                         'bots=bots_transfer.__main__:cli']
    # }
)
