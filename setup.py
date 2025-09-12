from setuptools import setup, find_packages

setup(
    name='smiles-encoder',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
	"chemprop>=2.2.1",
	"transformers>=4.56.0",
	"torch>=2.7.1",
	"huggingface-hub>=0.34.4",
	"numpy>=2.3.2"
    ],
    author='Jack Norris, Fanni Ruiz, Malena Russo',
    author_email='jacktimothynorris@gmail.com',
    python_requires=">=3.13",
    description='Utility class for encoding SMILES',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jacktnorris/smiles-encoder',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
