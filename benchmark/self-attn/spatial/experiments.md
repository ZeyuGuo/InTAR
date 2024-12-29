Problem:
Synthesis Hangs whem doing tapac compile. 

# Small Size Compile
1. N = 32, D = 128, Success, but it's weird that 0 DSP is used.
```
The total area of the design:
  BRAM: 51 / 3504 = 1.5%
  DSP: 0 / 8496 = 0.0%
  FF: 77028 / 2331840 = 3.3%
  LUT: 83109.125 / 1165920 = 7.1%
  URAM: 0 / 960 = 0.0%
```

2. N = 64, D = 256, Success, uses some DSP.
```
The total area of the design:
  BRAM: 19 / 3504 = 0.5%
  DSP: 11 / 8496 = 0.1%
  FF: 93040 / 2331840 = 4.0%
  LUT: 117080.125 / 1165920 = 10.0%
  URAM: 0 / 960 = 0.0%
```

# Vitis HLS for Individual Components
