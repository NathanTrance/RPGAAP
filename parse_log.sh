#!/bin/bash
LOGFILE="$1"
echo "$LOGFILE"
grep "\{'f/tst_" "$LOGFILE" | tail -1
