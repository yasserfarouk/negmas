#!/usr/bin/env bash
source .env
pytest --cov=negmas negmas tests
coverage xml
python-codacy-coverage -r coverage.xml
