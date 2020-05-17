#!/bin/bash
#submitjob -p MINI -q LONGGPU  -o -l longjob=1 -l hostname="node20|node26|node25|node24|node23" -eo $1 $2
#submitjob -p MINI -q GPU  -o  -l hostname="node20|node25|node26|node23|node24" -eo $1 $2
#submitjob -p MINI -q GPU  $1 ./run.sh
#submitjob -p MINI -q NORMAL -m 100000 $1 ./run.sh
submitjob -p MINI -q LONG    -o -l longjob=1 -l hostname="node11|node12|node13|node14|node15|node16|node17|node18|node19" -eo  -m 100000 $1 $2
