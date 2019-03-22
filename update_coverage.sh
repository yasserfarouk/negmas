#!/usr/bin/env bash
export CODACY_PROJECT_TOKEN=8ea7e86796c243f79d654de43458ed73
coverage xml
python-codacy-coverage -r coverage.xml


