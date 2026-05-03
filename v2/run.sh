#!/bin/sh -e

SQUARE=Rh5

parallel \
    --ssh 'ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ConnectTimeout=15 -o BatchMode=yes' \
    --sshloginfile workers.txt \
    --transferfile daq-$SQUARE \
    --joblog $SQUARE.log --retries 3 --resume-failed \
    --results $SQUARE.d/ \
    --eta \
    -- ./daq-$SQUARE {1} :::: <(seq 0 0x200)
