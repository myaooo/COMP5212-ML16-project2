#!/bin/bash

ssh mingyao@vis.cse.ust.hk "cd ~/ml16/COMP5212-ML16-project2/code; git pull; bash ./run.sh"
scp mingyao@vis.cse.ust.hk:~/ml16/COMP5212-ML16-project2/code/output/* ./output