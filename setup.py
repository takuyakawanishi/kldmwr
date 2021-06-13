from setuptools import setup, find_packages


setup(
    name='kldmwr',  # Required
    version='0.6.4a',  # Required
    description="Parameter Estimation Library Based on KLDMWR Framework",
    url='https://github.com/takuyakawanishi/kldmwr',
    author='Takuya Kawanishi',
    author_email='kawanishi@se.kanazawa-u.ac.jp',
    keywords='maximum likelihood, maximum product of spacings, '
             'maximum spacings, Kullback-Leibler divergence',
    license='MIT',
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'scipy', 'pandas'],
)
