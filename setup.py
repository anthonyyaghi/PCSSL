"""
DBBD: Dual Branch Bi-Directional Self-Supervised Learning for Point Clouds
Phase 2: Data Integration & Feature Processing
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='dbbd',
    version='0.2.0',
    description='Dual Branch Bi-Directional Self-Supervised Learning for Point Clouds - Phase 2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DBBD Team',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/DBBD',
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
        ],
        'vis': [
            'matplotlib>=3.5.0',
            'open3d>=0.15.0',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='point-cloud self-supervised-learning 3d-vision deep-learning',
)
