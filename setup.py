#!/usr/bin/env python3


from distutils.core import setup, Extension

setup(name='simlib',
      version='in-dev',
      description='Python Simulation Library',
      author='Zhen NI',
      author_email='z.ni@hotmail.com',
      packages=['simlib', 'simlib.blocks', 'simlib.tools', 'simlib.blocks.special'],
      package_data={'simlib': ['simulinkplot.m']},
     )
