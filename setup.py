# -*- coding: utf-8 -*-

import os
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

if __name__ == '__main__':
    from distutils.core import setup 
    extra_files = package_files('DesOptPy')
    setup(name='DesOptPy',
          version='1.3pre',
          description='DESign OPTimization in PYthon',
          author='E. J. Wehrle',
          author_email='Erich.Wehrle@unibz.it',
          copyright="Copyright 2015, 2016, E. J. Wehrle",
          package_data={'': extra_files},
          license='GNU Lesser General Public License',
          url='www.DesOptPy.org',
          packages=['DesOptPy'])
