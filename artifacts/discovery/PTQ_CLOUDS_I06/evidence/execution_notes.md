# Execution notes

Two substantive calculation cycles: (1) one rank-one multiplier deformation, exact coframe/scalar reconstruction and auxiliary connection elimination; (2) physical/constraint/probe tests and falsification. No second model, potential tuning, or physical revision.

The initial dynamics-script invocation stopped on a bookkeeping comparison: Python division produced `3.0`, and SymPy's exact `0` comparison did not identify residual `0.0` as the exact integer zero. The final script keeps exact integer arithmetic. The count was additionally tied to the explicit seven-velocity Hessian and four-constraint matrix, instead of a stand-alone arithmetic count. No physical coefficient or conclusion was changed. An initial lexical warning (`**2for` without separating whitespace) in the geometry script was corrected; that first geometry calculation had passed its checks. These are implementation corrections, not hidden failed physical candidates.
