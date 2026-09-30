"""SAR oil-spill detection -> drift backtracking -> AIS vessel attribution (SIH 26143).

Stages
------
1. detection   : Otsu baseline vs. trained DeepLabV3+ (.pt bundle), benchmarked and auto-selected
2. forcing     : wind + surface current (hard-coded demo values OR real GRIB/CSV/NetCDF)
3. backtrack   : Lagrangian ensemble (NumPy) with optional PyGNOME cross-check
4. attribution : time-synchronous AIS matching, anomaly scoring, dark-gap flags
5. twin        : forward simulation (NumPy or OpenOil) of each suspect's release
6. fusion      : ranked suspects with explicit, non-accusatory confidence levels
"""
__version__ = "1.0.0"
