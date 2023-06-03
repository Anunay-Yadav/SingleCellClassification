from setuptools import setup,find_packages
import sys, os

setup(name="NFSCC",
      description="flows on single cell classification data",
      version='0.1',
      author='Anunay',
      author_email='anunay18021@iiitd.ac.in',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml'],
      packages=find_packages(),
)

import sys  
sys.path.insert(0, '')
