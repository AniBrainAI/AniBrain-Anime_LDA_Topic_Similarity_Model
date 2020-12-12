import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as r:
    require_list = r.read().strip().split("\n")

setuptools.setup(
    name="anime_recs_synopsis",
    version="0.0.01",
    author="Chidi Udeze",
    author_email="chidiu98@gmail.com",
    description="Anime recommender model which measures synopsis/genre similarity to generate recommendations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/koji98/AniBrain-Anime_LDA_Topic_Similarity_Model",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=require_list,  # external packages as dependencies
)