#!/bin/bash

python -m black ovtrack
python -m isort ovtrack
python -m pylint ovtrack
python -m pydocstyle ovtrack
python -m mypy --strict ovtrack