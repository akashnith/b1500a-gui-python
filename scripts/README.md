# Double-click scripts (Windows)

Each one changes to the repository root first, so it works from anywhere.

| script | what it does |
|---|---|
| `gui.bat` | launch the graphical interface |
| `check.bat` | diagnose the VISA / WGFMU connection |
| `demo.bat` | a simulated WGFMU LTP/LTD run, no instrument needed |
| `smu.bat` | a simulated bipolar resistive-switching I-V, no instrument needed |
| `test.bat` | run the test suite |
| `b1500.bat` | forward any CLI command, e.g. `b1500 smu list` |

On Linux or macOS, call the module directly instead: `python -m b1500_studio gui`.
