# GCN_Nsys结果
- torch
~~~text
[1/8] [========================100%] torch_gcn_profile_nsys.nsys-rep
[2/8] [========================100%] torch_gcn_profile_nsys.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/CodeInstrumentation/TestCase_GCN/res/TorchRes/torch_gcn_profile_nsys.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
     48.3    1,602,579,192         28   57,234,971.1   78,157,206.5        2,730  100,161,327  45,977,068.9  poll
     45.3    1,500,221,586          3  500,073,862.0  500,071,092.0  500,068,942  500,081,552       6,745.9  pthread_cond_timedwait
      2.3       77,528,675      1,734       44,710.9       26,515.0        1,000    7,979,719     265,843.3  ioctl
      1.9       63,197,473      1,493       42,329.2        2,250.0        1,000   24,512,410     754,061.4  read
      1.9       62,655,938      2,013       31,125.7        2,780.0        1,000   15,742,267     470,366.7  open64
      0.1        1,779,854        401        4,438.5        2,300.0        1,000      262,511      13,655.5  mmap64
      0.0        1,274,416        255        4,997.7        2,520.0        1,020      286,491      18,507.6  munmap
      0.0        1,186,944          1    1,186,944.0    1,186,944.0    1,186,944    1,186,944           0.0  fork
      0.0          979,236         17       57,602.1       58,991.0       52,840       59,761       2,252.3  sleep
      0.0          757,303         15       50,486.9       10,970.0        2,440      243,951      70,443.7  pthread_join
      0.0          582,992         27       21,592.3        2,790.0        1,080      267,051      58,240.8  fopen
      0.0          531,862          1      531,862.0      531,862.0      531,862      531,862           0.0  waitpid
      0.0          463,343         70        6,619.2        5,445.0        1,480       25,170       4,448.7  mmap
      0.0          402,800         10       40,280.0       27,270.0        6,120      181,990      52,548.8  sem_timedwait
      0.0          392,101         12       32,675.1        2,995.0        1,070      348,431      99,479.2  open
      0.0          380,033         12       31,669.4       32,140.0       20,060       46,261       9,410.8  pthread_create
      0.0          250,381         20       12,519.1       11,440.0        2,440       30,990       6,829.1  fgets
      0.0           46,251         13        3,557.8        2,491.0        1,000        9,260       2,572.9  write
      0.0           27,850          3        9,283.3       11,830.0        2,520       13,500       5,916.4  pthread_mutex_lock
      0.0           18,640          4        4,660.0        3,545.0        1,440       10,110       3,886.7  fopen64
      0.0           15,390         10        1,539.0        1,280.0        1,020        3,480         757.2  fclose
      0.0           12,150          3        4,050.0        4,560.0        2,130        5,460       1,722.6  pipe2
      0.0            9,800          7        1,400.0        1,110.0        1,040        3,090         748.6  pthread_cond_signal
      0.0            8,050          3        2,683.3        2,460.0        1,080        4,510       1,725.9  fread
      0.0            6,770          2        3,385.0        3,385.0        1,780        4,990       2,269.8  socket
      0.0            4,830          1        4,830.0        4,830.0        4,830        4,830           0.0  connect
      0.0            1,610          1        1,610.0        1,610.0        1,610        1,610           0.0  fputs_unlocked
      0.0            1,160          1        1,160.0        1,160.0        1,160        1,160           0.0  fcntl
      0.0            1,140          1        1,140.0        1,140.0        1,140        1,140           0.0  bind

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)    Min (ns)   Max (ns)     StdDev (ns)               Name
 --------  ---------------  ---------  -------------  ------------  --------  -----------  -------------  ----------------------------
     62.8      929,629,185        105    8,853,611.3       5,000.0     2,670  929,052,953   90,665,743.8  cudaMemcpyAsync
     28.5      422,289,647          3  140,763,215.7  74,223,282.0       580  348,065,785  183,324,899.6  cudaFree
      8.5      125,606,630          4   31,401,657.5       1,230.0       690  125,603,480   62,801,215.0  cudaStreamIsCapturing_v10000
      0.1        1,012,603         69       14,675.4       4,810.0     3,520      637,033       76,071.2  cudaLaunchKernel
      0.0          572,412         12       47,701.0       5,585.0     3,120      512,572      146,405.6  cudaHostAlloc
      0.0          542,094          8       67,761.8      78,680.0     3,320      108,211       38,802.5  cudaMalloc
      0.0          504,860         87        5,803.0       4,670.0     1,580       76,270        7,757.5  cudaStreamSynchronize
      0.0           74,811        365          205.0         180.0        70        1,060          107.3  cuGetProcAddress
      0.0           15,150          1       15,150.0      15,150.0    15,150       15,150            0.0  cudaMemcpy
      0.0           12,910         29          445.2         320.0       230        1,780          374.8  cudaEventCreateWithFlags
      0.0            6,900         11          627.3         490.0       410        1,980          451.4  cudaEventRecord
      0.0            2,710          3          903.3         900.0       750        1,060          155.0  cuInit

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     16.0           44,287         11   4,026.1   4,096.0     2,976     4,799        495.3  void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void …
     15.4           42,720          3  14,240.0  14,240.0    14,240    14,240          0.0  ampere_sgemm_64x32_sliced1x4_tn
     10.8           29,857          8   3,732.1   3,376.0     3,232     4,992        644.3  void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::MeanOps<fl…
      8.5           23,425          6   3,904.2   3,744.5     3,200     4,800        770.0  void at::native::batch_norm_transform_input_channels_last_kernel<float, float, float, (int)4>(const…
      6.4           17,857          6   2,976.2   3,104.5     2,432     3,328        372.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::…
      6.4           17,824          3   5,941.3   5,728.0     5,664     6,432        426.1  void at::native::<unnamed>::CatArrayBatchedCopy<float, unsigned int, (int)3, (int)64, (int)64>(T1 *…
      5.4           14,848          3   4,949.3   4,704.0     4,704     5,440        424.9  void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapp…
      5.1           14,144          5   2,828.8   2,816.0     2,784     2,880         36.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::softplus_kernel(at::TensorIterat…
      5.0           13,824          6   2,304.0   2,272.0     2,240     2,464         81.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::batch_norm_calc_invst…
      3.6           10,111          3   3,370.3   3,327.0     3,296     3,488        103.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::…
      3.4            9,344          3   3,114.7   3,040.0     3,040     3,264        129.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::…
      2.8            7,648          3   2,549.3   2,464.0     2,464     2,720        147.8  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, floa…
      2.3            6,431          1   6,431.0   6,431.0     6,431     6,431          0.0  ampere_sgemm_32x32_sliced1x4_tn
      2.3            6,336          3   2,112.0   2,080.0     2,048     2,208         84.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::deta…
      1.8            5,024          1   5,024.0   5,024.0     5,024     5,024          0.0  void gemmSN_TN_kernel_64addr<float, (int)128, (int)16, (int)2, (int)4, (int)8, (int)9, (bool)0, cub…
      1.6            4,352          1   4,352.0   4,352.0     4,352     4,352          0.0  void at::native::<unnamed>::CatArrayBatchedCopy<float, unsigned int, (int)2, (int)128, (int)1>(T1 *…
      1.3            3,616          1   3,616.0   3,616.0     3,616     3,616          0.0  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<fl…
      1.2            3,263          1   3,263.0   3,263.0     3,263     3,263          0.0  void gemv2T_kernel_val<int, int, float, float, float, (int)128, (int)16, (int)2, (int)4, (bool)0, (…
      0.8            2,304          1   2,304.0   2,304.0     2,304     2,304          0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::mse_kernel_cuda(at::TensorIterat…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     89.2          138,432     97   1,427.1     896.0       832     9,759      1,550.7  [CUDA memcpy Host-to-Device]
      8.4           13,087      6   2,181.2   2,176.0     2,143     2,272         47.3  [CUDA memcpy Device-to-Device]
      2.4            3,681      3   1,227.0   1,313.0     1,024     1,344        176.5  [CUDA memcpy Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      0.846     97     0.009     0.000     0.000     0.154        0.026  [CUDA memcpy Host-to-Device]
      0.002      6     0.000     0.000     0.000     0.001        0.000  [CUDA memcpy Device-to-Device]
      0.000      3     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]
~~~

- paddle
~~~text
[1/8] [========================100%] paddle_gcn_profile_nsys.nsys-rep
[2/8] [========================100%] paddle_gcn_profile_nsys.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool/CodeInstrumentation/TestCase_GCN/res/PaddleRes/paddle_gcn_profile_nsys.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)    Max (ns)     StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  ----------  -----------  -------------  ----------------------
     39.9    4,009,727,563          5  801,945,512.6  999,879,665.0  10,081,969  999,984,770  442,665,180.8  usleep                
     24.2    2,432,338,202         38   64,008,900.1  100,139,067.0       2,370  100,154,736   45,661,067.3  poll                  
     19.9    2,000,454,237          5  400,090,847.4  500,087,836.0     124,374  500,090,335  223,588,056.1  pthread_cond_timedwait
     11.9    1,193,535,706          4  298,383,926.5  298,810,578.5      25,499  595,889,050  343,140,845.0  pthread_cond_wait     
      1.4      139,293,880         24    5,803,911.7        2,935.0       1,000   45,314,626   11,742,312.2  fread                 
      1.1      108,904,433      2,291       47,535.8       25,929.0         999    9,434,060      329,584.1  ioctl                 
      1.0       99,788,804      2,604       38,321.4        2,030.0         999   30,356,410      947,166.4  read                  
      0.2       25,101,321          4    6,275,330.3    6,219,793.5       2,569   12,659,165    7,243,430.4  futex                 
      0.2       21,742,212         16    1,358,888.3    1,399,853.5       1,780    2,514,980      937,996.1  pthread_mutex_lock    
      0.1        8,804,180      3,435        2,563.1        2,070.0       1,050       30,128        1,393.5  open64                
      0.0        4,082,645          4    1,020,661.3    1,163,335.0     489,986    1,265,989      358,700.3  fork                  
      0.0        2,300,577        636        3,617.3        2,210.0       1,000      245,888       10,568.9  mmap64                
      0.0        1,419,743        388        3,659.1        2,330.0       1,000       41,418        4,321.4  munmap                
      0.0        1,299,920         16       81,245.0        8,770.0       4,950      379,022      127,254.1  pthread_join          
      0.0        1,017,615         57       17,852.9        3,840.0       1,050      359,372       64,189.7  fopen                 
      0.0          999,571         17       58,798.3       59,337.0      53,178       64,607        2,496.9  sleep                 
      0.0          546,954          4      136,738.5       19,584.5      14,669      493,116      237,612.7  waitpid               
      0.0          491,525         10       49,152.5       39,278.0      10,619      229,269       65,212.9  sem_timedwait         
      0.0          437,500         54        8,101.9        4,740.0       1,360       55,247       10,511.2  mmap                  
      0.0          365,520         12       30,460.0        6,164.5       1,570      297,126       84,046.8  open                  
      0.0          277,866          1      277,866.0      277,866.0     277,866      277,866            0.0  sem_wait              
      0.0          250,326         20       12,516.3       11,484.5       1,190       31,998        7,072.0  fgets                 
      0.0          249,418          8       31,177.3       32,033.5      19,759       39,208        7,471.0  pthread_create        
      0.0           83,073         37        2,245.2        1,650.0       1,000        6,909        1,422.9  fclose                
      0.0           54,048         16        3,378.0        2,385.0       1,060       10,979        2,682.2  write                 
      0.0           44,429         23        1,931.7        1,330.0       1,040        4,500        1,006.0  pthread_cond_signal   
      0.0           33,518         10        3,351.8        2,940.0       1,450        5,869        1,641.1  pipe2                 
      0.0           18,930          4        4,732.5        3,780.0       1,380        9,990        3,714.4  fopen64               
      0.0           14,370          9        1,596.7        1,380.0       1,110        2,640          504.4  fcntl                 
      0.0            9,648          3        3,216.0        2,969.0       1,260        5,419        2,090.5  fwrite                
      0.0            8,839          2        4,419.5        4,419.5       1,739        7,100        3,790.8  socket                
      0.0            7,909          3        2,636.3        3,339.0       1,200        3,370        1,244.0  pthread_cond_broadcast
      0.0            5,810          1        5,810.0        5,810.0       5,810        5,810            0.0  connect               
      0.0            2,890          1        2,890.0        2,890.0       2,890        2,890            0.0  putc                  
      0.0            2,340          1        2,340.0        2,340.0       2,340        2,340            0.0  prctl                 
      0.0            1,470          1        1,470.0        1,470.0       1,470        1,470            0.0  fputs_unlocked        
      0.0            1,080          1        1,080.0        1,080.0       1,080        1,080            0.0  sigaction             
      0.0            1,000          1        1,000.0        1,000.0       1,000        1,000            0.0  fflush                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)    Max (ns)     StdDev (ns)               Name            
 --------  ---------------  ---------  ------------  -----------  --------  -------------  ------------  ----------------------------
     61.1    1,064,588,721        200   5,322,943.6      3,740.0     2,470  1,062,987,850  75,164,024.4  cudaMalloc                  
     24.1      421,006,852         14  30,071,918.0        590.0       290    347,251,785  93,381,745.3  cudaFree                    
      7.8      135,706,498         16   8,481,656.1      1,410.0       950    135,532,896  33,880,352.0  cudaStreamCreateWithFlags   
      6.2      107,386,296        234     458,915.8    104,165.0     2,090      7,859,905     934,948.8  cuModuleUnload              
      0.6       10,401,254          2   5,200,627.0  5,200,627.0   534,075      9,867,179   6,599,501.1  cudaHostAlloc               
      0.1        1,714,872        261       6,570.4      3,989.0     3,309         29,429       4,595.0  cudaMemcpy                  
      0.1        1,520,093        136      11,177.2      6,149.0     3,900        579,893      49,248.4  cudaLaunchKernel            
      0.0          632,560         40      15,814.0     13,779.0    13,489         42,248       5,818.4  cudaMemGetInfo              
      0.0          370,671         56       6,619.1      5,690.0     3,769         21,949       3,390.7  cudaMemcpyAsync             
      0.0           78,495        384         204.4        180.0        70          1,000         104.8  cuGetProcAddress            
      0.0           29,397         73         402.7        330.0       240          1,360         219.9  cudaEventCreateWithFlags    
      0.0           19,129          1      19,129.0     19,129.0    19,129         19,129           0.0  cudaMemsetAsync             
      0.0           16,378          7       2,339.7      1,710.0     1,179          4,599       1,518.2  cudaStreamSynchronize       
      0.0            8,910          1       8,910.0      8,910.0     8,910          8,910           0.0  cudaStreamCreateWithPriority
      0.0            3,310          3       1,103.3        930.0       900          1,480         326.5  cuInit                      
      0.0            2,660          1       2,660.0      2,660.0     2,660          2,660           0.0  cudaEventRecord             
      0.0            1,320          1       1,320.0      1,320.0     1,320          1,320           0.0  cudaStreamIsCapturing_v10000

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     14.0           64,892          6  10,815.3  10,815.5     3,807    17,855      7,664.9  void phi::BNForwardInference<float, (common::DataLayout)2>(const T1 *, const phi::backends::gpu::Cu…
     12.9           59,869         30   1,995.6   1,952.0     1,920     2,272         90.3  void phi::funcs::VectorizedElementwiseKernel<float, phi::FullFunctor<float, float>, (int)0, (int)1,…
      7.6           35,069         16   2,191.8   2,112.0     2,079     2,624        168.5  void phi::funcs::VectorizedElementwiseKernel<int, phi::ScaleFunctor<int, int>, (int)1, (int)1, (int…
      7.2           33,245          3  11,081.7  11,199.0    10,655    11,391        381.8  void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align1>(T1::Params)                
      6.4           29,569         11   2,688.1   2,624.0     2,304     3,168        269.6  void phi::funcs::ReduceHigherDimKernel<float, float, float, phi::kps::AddFunctor<float>, phi::kps::…
      6.0           28,062          3   9,354.0   9,503.0     8,608     9,951        683.8  void phi::funcs::GatherNdCUDAKernel<float, int>(const T1 *, common::Dim<(int)9>, const T2 *, T1 *, …
      5.5           25,597          8   3,199.6   3,072.0     2,784     4,063        448.4  void phi::funcs::VectorizedElementwiseKernel<float, phi::funcs::CudaSoftplusFunctor<float>, (int)1,…
      5.5           25,504          8   3,188.0   3,184.0     3,136     3,264         45.1  void phi::funcs::GatherNdCUDAKernel<float, long>(const T1 *, common::Dim<(int)9>, const T2 *, T1 *,…
      5.2           24,060          8   3,007.5   2,879.5     2,336     3,743        592.1  void phi::funcs::VectorizedBroadcastKernel<phi::funcs::AddFunctor<float>, float, (int)2, (int)1, (i…
      4.1           19,005          3   6,335.0   6,207.0     6,143     6,655        279.0  void phi::funcs::ConcatTensorWithDifferentShape<int, (int)4, phi::funcs::PointerAndColWrapper<float…
      3.3           15,135          6   2,522.5   2,608.0     2,272     2,688        187.3  void phi::funcs::DistributionKernel<float, phi::funcs::uniform_distribution<float>, phi::funcs::uni…
      2.7           12,736          3   4,245.3   4,160.0     4,064     4,512        235.9  void phi::funcs::SplitTensorWithDifferentShape<float, int, phi::funcs::PointerArray<float, (phi::fu…
      2.4           11,263          2   5,631.5   5,631.5     5,247     6,016        543.8  void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(T1::Params)                
      1.9            8,768          3   2,922.7   2,848.0     2,848     3,072        129.3  void phi::funcs::VectorizedElementwiseKernel<float, phi::funcs::CudaSigmoidFunctor<float>, (int)1, …
      1.8            8,543          3   2,847.7   2,848.0     2,783     2,912         64.5  void phi::funcs::VectorizedBroadcastKernel<phi::kps::IdentityFunctor<float, float>, float, (int)1, …
      1.8            8,256          1   8,256.0   8,256.0     8,256     8,256          0.0  void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_tt_align1>(T1::Params)                
      1.8            8,126          3   2,708.7   2,719.0     2,687     2,720         18.8  void phi::funcs::VectorizedBroadcastKernel<phi::funcs::MultiplyFunctor<float>, float, (int)2, (int)…
      1.6            7,232          3   2,410.7   2,400.0     2,272     2,560        144.3  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<float, float, int, phi::kps::AddFunc…
      1.4            6,657          3   2,219.0   2,209.0     2,208     2,240         18.2  void phi::funcs::VectorizedBroadcastKernel<phi::funcs::AddFunctor<float>, float, (int)2, (int)1, (i…
      1.4            6,464          3   2,154.7   2,080.0     2,048     2,336        157.9  void phi::funcs::VectorizedElementwiseKernel<float, phi::ScaleFunctor<float, float>, (int)1, (int)1…
      1.2            5,504          2   2,752.0   2,752.0     2,432     3,072        452.5  void phi::funcs::StackCudaKernel<float, int, phi::funcs::ConstPointerArray<float, (phi::funcs::Segm…
      1.1            5,120          2   2,560.0   2,560.0     2,336     2,784        316.8  void phi::funcs::VectorizedBroadcastKernel<phi::funcs::MultiplyFunctor<float>, float, (int)2, (int)…
      1.0            4,767          2   2,383.5   2,383.5     2,271     2,496        159.1  void phi::funcs::VectorizedBroadcastKernel<phi::funcs::SubtractFunctor<float>, float, (int)2, (int)…
      0.7            3,199          1   3,199.0   3,199.0     3,199     3,199          0.0  void phi::funcs::VectorizedBroadcastKernel<phi::funcs::DivideFunctor<float, void>, float, (int)2, (…
      0.6            2,944          1   2,944.0   2,944.0     2,944     2,944          0.0  void phi::funcs::VectorizedBroadcastKernel<phi::funcs::SubtractFunctor<float>, float, (int)2, (int)…
      0.5            2,144          1   2,144.0   2,144.0     2,144     2,144          0.0  void phi::funcs::VectorizedElementwiseKernel<float, phi::CudaAbsFunctor<float, void>, (int)1, (int)…
      0.5            2,144          1   2,144.0   2,144.0     2,144     2,144          0.0  void phi::funcs::VectorizedElementwiseKernel<float, phi::funcs::CudaSquareFunctor<float>, (int)1, (…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     61.2          278,425    218   1,277.2     896.0       832     6,143      1,207.8  [CUDA memcpy Host-to-Device]  
     25.2          114,684     56   2,047.9   2,016.0     1,951     2,335         94.9  [CUDA memcpy Device-to-Device]
     13.0           59,161     43   1,375.8   1,056.0     1,023     4,543        878.5  [CUDA memcpy Device-to-Host]  
      0.5            2,432      1   2,432.0   2,432.0     2,432     2,432          0.0  [CUDA memset]                 

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      1.490    218     0.007     0.000     0.000     0.087        0.021  [CUDA memcpy Host-to-Device]  
      0.374     56     0.007     0.002     0.000     0.059        0.012  [CUDA memcpy Device-to-Device]
      0.329     43     0.008     0.001     0.000     0.087        0.023  [CUDA memcpy Device-to-Host]  
      0.009      1     0.009     0.009     0.009     0.009        0.000  [CUDA memset]                 
~~~


# VGNN_Nsys结果


# nvtx与nsight配合
~~~bash
pip install nvtx -i https://pypi.douban.com/simple/
~~~


~~~python
import nvtx
import torch
from datetime import datetime

warmup_time = 3
test_time = 5
# warmup
for kk in range(warmup_time):
    noise_pred = self.transformer(*input).sample
# for ncu
torch.cuda.synchronize(0)
transformer_nvtx = nvtx.start_range(message="transformer", color="blue")

noise_pred = self.transformer(*input).sample

torch.cuda.synchronize(0)
nvtx.end_range(transformer_nvtx)
# for cost_time
torch.cuda.synchronize(0)
start_time = datetime.now()
# repeat
for kk in range(test_time):
    noise_pred = self.transformer(*input).sample

torch.cuda.synchronize(0)
end_time = datetime.now()
during_time = end_time - start_time
time_ms = during_time.seconds * 1000 + during_time.microseconds / 1000.0
msg = f"The whole time:  {time_ms/test_time} ms\n"
print(msg)
with open(file_path, "w") as time_cost_file:
    time_cost_file.write(msg)
~~~


~~~bash
sudo /usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "transformer" \
--metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__grid_size,\
launch__block_size,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
sm__warps_active.avg.pct_of_peak_sustained_active \
--csv python demo3.py > metrics_713_paddle_.csv 
~~~