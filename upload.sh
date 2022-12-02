#!/bin/bash
rm ./dist/*.gz
python setup.py sdist
twine upload dist/*.gz