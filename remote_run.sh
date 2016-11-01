#!/bin/bash

ssh mingyao@vis.cse.ust.hk "cd ~/ml16/COMP5212-ML16-project2/code; git pull; bash ./run.sh"
scp -r mingyao@vis.cse.ust.hk:~/ml16/COMP5212-ML16-project2/code/output/ ./output
scp -r mingyao@vis.cse.ust.hk:~/ml16/COMP5212-ML16-project2/code/run.log ./