# ns-3 local setup

This workspace is prepared for a local `ns-3.47` setup on macOS.

## What is installed here

- `./ns-3.47`: official ns-3 source tree
- `./.tools`: local Python-based build tools (`cmake`, `ninja`)
- `./ns3-env.zsh`: small helper script to expose the local tools in your shell

## First run

Open a terminal in this folder and run:

```zsh
source ./ns3-env.zsh
cd "$NS3_HOME"
./ns3 run hello-simulator
```

If you want to rebuild manually:

```zsh
source ./ns3-env.zsh
cd "$NS3_HOME"
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

## Where to write your own simulation

The easiest place to start is:

- `./ns-3.47/scratch/`

For example:

```zsh
cd ./ns-3.47
cp examples/tutorial/first.cc scratch/my-first.cc
./ns3 run scratch/my-first
```

## Capstone starter scenario

There is also a starter file for your current capstone task:

```zsh
source ./ns3-env.zsh
cd "$NS3_HOME"
./ns3 run scratch/uav-adhoc-logging
```

Useful variations:

```zsh
./ns3 run "scratch/uav-adhoc-logging --numUavs=3 --spacing=20 --simTime=15"
./ns3 run "scratch/uav-adhoc-logging --numUavs=5 --spacing=35 --speed=2.5"
```

This starter scenario prints:

- UAV positions
- RSSI samples
- RTT samples
- PLR summary

To generate an interactive HTML report from the CSV files:

```zsh
python3 /Users/inyoung/Desktop/캡디1/ns3/plot_uav_report.py
```

## Notes

- This setup keeps tools local to the workspace instead of changing the whole system.
- Some optional ns-3 features were skipped because extra libraries are not installed yet.
- Typical first experiments like `scratch/` examples and core tutorials work fine without them.
