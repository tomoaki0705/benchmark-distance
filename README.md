benchmark-distance
==================

This code provides a benchmark for hardware-accelerated distance computation such as L2 (Euclidean), L1 (Manhattan), and Hamming distance. Now it support distance between vectors with byte (uchar) elements only.

###Results of our benchmark

#### Test environment
* **CPU**: Intel Core i5 M560 @ 2.67GHz x2 (Nehalem)
* **Memory**: 8GB
* **IDE**: Visual Studio 2010 Professioal
* **Platform**: x64 
* **Option**: /O2

#### (D=128 and N=4M w/o SSE)
* **L2 (Euclidean)**: 461ms
* **L1 (Manhattan)**: 586ms
* **Hamming 32bits**: 841ms
* **Hamming 64bits**: 527ms

#### (D=128 and N=4M w/ SSE)
* **L2 (Euclidean)**: 129ms
* **L1 (Manhattan)**:  91ms
* **Hamming 32bits**: 189ms
* **Hamming 64bits**: 128ms

