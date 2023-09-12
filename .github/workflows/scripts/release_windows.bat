echo on

python -m pip install --upgrade pip
pip install setuptools wheel twine auditwheel

pip wheel . -w wheel/ --no-deps
twine upload --skip-existing wheel/*
