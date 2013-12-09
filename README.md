benchmark-distance
==================

This code provides a benchmark for hardware-accelerated distance computation such as L2 (Euclidean), L1 (Manhattan), and Hamming distance. Now it support distance between vectors with byte (uchar) elements only.

### Results of our benchmark

#### Benchmark. 1
**Test environment**
* **CPU**: Intel Core i7 4770 CPU @ 3.40GHz x4 (Haswell)
* **Memory**: 8.0 GB
* **IDE**: Visual Studio 2012 Professioal
* **Platform**: x64 
* **Option**: /O2

**Results (D=128 and N=4M)**
* **L2 (Euclidean)**: 313ms (w/o SSE),  52ms (w/ SSE)
* **L1 (Manhattan)**: 291ms (w/o SSE),  33ms (w/ SSE)
* **Hamming 32bits**: 275ms (w/o SSE),  62ms (w/ SSE)
* **Hamming 64bits**: 192ms (w/o SSE),  43ms (w/ SSE)

#### Benchmark. 2
**Test environment**
* **CPU**: Intel Core i5 560M @ 2.67GHz x2 (Arrandale)
* **Memory**: 8.0 GB
* **IDE**: Visual Studio 2010 Professioal
* **Platform**: x64 
* **Option**: /O2

**Result (D=128 and N=4M)**
* **L2 (Euclidean)**: 461ms (w/o SSE), 127ms (w/ SSE)
* **L1 (Manhattan)**: 576ms (w/o SSE),  90ms (w/ SSE)
* **Hamming 32bits**: 708ms (w/o SSE), 164ms (w/ SSE)
* **Hamming 64bits**: 461ms (w/o SSE), 108ms (w/ SSE)
