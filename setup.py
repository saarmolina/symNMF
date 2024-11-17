from setuptools import setup, Extension

symnmf_module = Extension('symnmf',
                         sources=['symnmfmodule.c', 'symnmf.c'])

setup(name='symnmf',
      version='1.0',
      description='Symmetric Non-negative Matrix Factorization implementation',
      ext_modules=[symnmf_module])