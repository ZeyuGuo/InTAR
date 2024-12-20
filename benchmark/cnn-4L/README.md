# 4-Layer CNN logs/measurements

- CSIM:
```
Cycle count: 24266
Kernel time (ns): 153645503
Kernel time (us): 153646
Kernel time (ms): 153.646
```

- Vitis HW EMU
```
Cycle count: 259748
Kernel time (ns): 149014639534
Kernel time (us): 1.49015e+08
Kernel time (ms): 149015
```

- Vitis HW
```
Cycle count: 259815
Kernel time (ns): 1129033
Kernel time (us): 1129.03
Kernel time (ms): 1.12903
```

BRAM/DSP Utilization:
```
+-------------------+-------+-------+------------+-----------+-------+
|     Site Type     |  Used | Fixed | Prohibited | Available | Util% |
+-------------------+-------+-------+------------+-----------+-------+
| Block RAM Tile    | 202.5 |     0 |          0 |      2016 | 10.04 |
|   RAMB36/FIFO*    |   198 |   196 |          0 |      2016 |  9.82 |
|     RAMB36E2 only |   198 |       |            |           |       |
|   RAMB18          |     9 |     8 |          0 |      4032 |  0.22 |
|     RAMB18E2 only |     9 |       |            |           |       |
| URAM              |     0 |     0 |          0 |       960 |  0.00 |
+-------------------+-------+-------+------------+-----------+-------+
```

```
+----------------+------+-------+------------+-----------+-------+
|    Site Type   | Used | Fixed | Prohibited | Available | Util% |
+----------------+------+-------+------------+-----------+-------+
| DSPs           |  871 |     4 |          0 |      9024 |  9.65 |
|   DSP48E2 only |  871 |       |            |           |       |
+----------------+------+-------+------------+-----------+-------+
```
