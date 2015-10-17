from distutils.core import setup
setup(
  name = 'DesOptPy',
  packages = ['DesOptPy'], # this must be the same as the name above
  package_data={'DesOptPy': ['ResultReportFiles/*', 'StatusReportFiles/*', 'Doc/*']},
  version = "1.0",
  description = 'DESign OPTimization in PYthon',
  url = "DesOptPy.org",
  author = "E. J. Wehrle",
  author_email = "wehrle(a)tum.de",
  classifiers = [],
)
