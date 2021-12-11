# !/usr/bin/env python

from setuptools import setup

requirements = [
    'opencv-python',
    'tensorflow==1.14.0',
    'matplotlib',
    'PyQt5',
]

setup_requirements = ['wheel']

test_requirements = [ ]

setup(
    name='Object Classification',
    packages= [],
    version='0.1.0',
    description='Application to run a object detection model (NN) on a images/video stream',
    install_requires=requirements,
    author='Noah Langat',
    license='MIT',
    author_email='knoah.lr@gmail.com',
    url='https://github.com/knoahlr',
    keywords=['tensorflow', 'object classification' ],
    python_requires='==3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development',
    ],
    setup_requires=setup_requirements,
    tests_require=test_requirements,
)
