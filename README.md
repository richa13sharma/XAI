# XAI

Explainable Artificial Intelligence for the Capstone Project


## Introduction

This repository holds our basic code from Phase 1 written in Python 3.

[WIP: Please add details in the form of sections as you add your code.]




## Setup

### Pre-requisites

- python3

### Steps

Use `venv` to ensure using the same version of packages found in `requirements.txt`

- [Create a virtual environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments)  
  `python3 -m venv .env`
- Activate virtual environment
  - Unix-like: `source .env/bin/activate`
  - Windows: `.env\Scripts\activate.bat`
- Install dependencies  
  `pip install -r requirements.txt`

> Note: Remember to activate venv and install dependencies after doing a `git pull`


## File Structure

- `data` - Holds data such as datasets, images etc.
- `obj` - Holds pickled files, etc.

- `main.py` - Run using `python3 main.py`

[WIP: Please add details in the form of sections as you add your code.]


## Code Guide

### Formatting and linting

Use Black as the code formatter. Functions internal to a class start with an underscore. Use docstrings to comment every class and function. Define additional functions that provide common utility in `utils.py`. Use underscores (`_`) to split all variables and functions (class names must begin with a capital letter). Multi-use imports at the start of file but inline imports preferred.

### Adding a new dependency

- Activate your virtual environment
- `pip install` dependency
- `pip freeze > requirements.txt`


## Version

- Python: 3.8.5