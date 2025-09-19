from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="scfp-framework",
    version="1.0.0",
    author="Shahed Almobydeen, Gaith Rjoub, Jamal Bentahar, Ahmad Irjoob",
    author_email="salmobydeen@aut.edu.jo",
    description="Self-Correction Failure Prediction Framework for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scfp-framework/scfp-impl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.23.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "scfp-train=scfp.cli:train",
            "scfp-evaluate=scfp.cli:evaluate",
            "scfp-predict=scfp.cli:predict",
            "scfp-route=scfp.cli:route",
        ],
    },
    include_package_data=True,
    package_data={
        "scfp": ["configs/*.yaml", "templates/*.txt"],
    },
)
