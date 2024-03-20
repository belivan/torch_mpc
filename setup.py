from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['torch_mpc', 'torch_mpc.models', 'torch_mpc.action_sampling'],
  package_dir={'': 'src'}
)

setup(**d)
