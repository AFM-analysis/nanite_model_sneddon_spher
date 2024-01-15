from setuptools import Extension, setup

import numpy as np


setup(
    ext_modules=[
        Extension(
            "nanite_model_sneddon_spher.model_sneddon_spherical",
            sources=[
                "src/nanite_model_sneddon_spher/model_sneddon_spherical.pyx"],
            include_dirs=[np.get_include()],
        )
    ]
)
