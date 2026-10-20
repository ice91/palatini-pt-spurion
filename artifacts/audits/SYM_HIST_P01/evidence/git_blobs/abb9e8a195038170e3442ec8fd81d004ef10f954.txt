import numpy as np
from palatini_pt.algebra import D, E, TdotDeps, T2, project_to_canonical, projection_matrix, CANONICAL_BASIS, EXTENDED_BASIS

def test_ibp_and_projection():
    vec = project_to_canonical(3*D + 2*E + 5*TdotDeps - 7*T2)
    assert np.all(vec == np.array([1, 5, -7], dtype=vec.dtype))
    P = projection_matrix(EXTENDED_BASIS, CANONICAL_BASIS)
    # 乘上 [0,1,0,0]^T 應得 [-1,0,0]^T
    eE = np.zeros((4,), dtype=P.dtype); eE[1] = 1
    assert np.all(P @ eE == np.array([-1,0,0], dtype=P.dtype))
