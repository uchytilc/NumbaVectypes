from numba import cuda
#import impl to update CUDA registry with vector functionality
import cudaimpl
import vectypeimpl

from vectype import ( char2,  short2,  int2,  long2,
                     uchar2, ushort2, uint2, ulong2,
                     float2, double2,
                      char3,  short3,  int3,  long3,
                     uchar3, ushort3, uint3, ulong3,
                     float3, double3)
import cudafuncs

import numpy as np

@cuda.jit()
def kernel(out):
    a = float2() + float2(1.0, 3)
    b = float3(float2(1,2), 3)
    c = int3(1)

    d = float3(1) * 3
    e = d.xz
    f = d.yz

    g = float3(1.2,2.1,3.6)
    h = cudafuncs.sum(g)
    i = cudafuncs.dot(g, g)
    j = cudafuncs.length(g)
    k = abs(g)

    l = cudafuncs.max(g, d)
    m = cudafuncs.max(g)
    n = cudafuncs.min(g, 5)

    o = cudafuncs.sin(g)

out = np.zeros(26, dtype = np.float32)

kernel[1, 1](out)
# print(out)
