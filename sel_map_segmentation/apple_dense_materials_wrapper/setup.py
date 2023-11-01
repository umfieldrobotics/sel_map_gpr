from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['apple_dense_materials_wrapper', 'apple_dense_materials'],
    package_dir={'': 'src'}
)
setup(**d)

