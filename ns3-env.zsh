#!/bin/zsh

SCRIPT_DIR=${0:A:h}
export NS3_HOME="$SCRIPT_DIR/ns-3.47"
export PATH="$SCRIPT_DIR/.tools/bin:$PATH"

echo "NS3_HOME=$NS3_HOME"
echo "PATH updated with $SCRIPT_DIR/.tools/bin"
echo "Try: cd \"$NS3_HOME\" && ./ns3 run hello-simulator"
