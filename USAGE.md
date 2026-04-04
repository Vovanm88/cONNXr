```
make all                                          # default (scalar tiled)                                                    
make all CONNXR_SIMD=sse4                         # + SSE4.1                                                                      
make all CONNXR_SIMD=avx2                         # + AVX2+FMA                                                                    
make all CONNXR_SIMD=avx2 CONNXR_OMP=1            # + OpenMP                                                                      
make all CONNXR_SIMD=avx2 CONNXR_OMP=1 CONNXR_BLAS=1  # BLAS backend                                                              
make benchmark_matmul CONNXR_SIMD=avx2 CONNXR_OMP=1    # operator benchmark         
```

bench only:
```
# Scalar tiled (no SIMD, no threads)
make benchmark_matmul                                                                                                             
# With SSE4.1                                                                                                                     
make benchmark_matmul CONNXR_SIMD=sse4                                                                                            
# With AVX2+FMA                                                                                                                   
make benchmark_matmul CONNXR_SIMD=avx2                                                                                            
# With AVX2+FMA + OpenMP                                                                                                          
make benchmark_matmul CONNXR_SIMD=avx2 CONNXR_OMP=1     
```