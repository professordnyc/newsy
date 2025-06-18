from setuptools import setup, find_packages

setup(
    name="newsy",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        # List your project's dependencies here
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "requests-cache>=0.9.1",
        "newspaper3k>=0.2.8",
        "lxml>=4.6.3",
        "google-search-results>=2.4.0",
        "pydantic>=1.8.0",
    ],
    python_requires=">=3.8",
)
