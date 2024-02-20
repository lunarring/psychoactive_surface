from setuptools import setup, find_packages

# Read requirements.txt and store its contents in a list
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='coupling_diffusion',
    version='0.1',
    url='https://github.com/lunarring/coupling_diffusion',
    description='Real-time diffusion coupled to external parameters',
    long_description=open('README.md').read(),
    install_requires=[
        'lunar_tools @ git+https://github.com/lunarring/lunar_tools.git#egg=lunar_tools'
    ] + required,
    include_package_data=False,
)


