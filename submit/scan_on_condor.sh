#!/bin/bash
export PYTHONPATH=/eos/user/f/fvaselli/piptarget/
FOLDER=$1
OTHER=$2

echo From based folder: $FOLDER
echo Other options: $OTHER
git clone git@github.com:francesco-vaselli/fAD.git
cd fAD/experiments
python3 ./hyperparameter_search.py
cp -r ./hyperparameter_search_results $FOLDER/.