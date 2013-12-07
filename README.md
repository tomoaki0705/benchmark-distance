benchmark-distance
==================

This code provides a benchmark for hardware-accelerated distance computation such as L2 (Euclidean), L1 (Manhattan), and Hamming distance. Now it support distance between vectors with byte (uchar) elements only.

Our benchmark environments:
* *CPU*: Intel Core i5 M560 @ 2.67GHz x2 (Nehalem)
* *Memory*: 8GB
* *IDE*: Visual Studio 2010 Professioal
* *Platform*: x64 
* *Option*: /O2

Our results:
(D=128 and N=4M w/o SSE)
* **L2**: 461ms
* **L1**: 586ms
* **Hm32**: 841ms
* **Hm64**: 527ms
(D=128 and N=4M w/o SSE)
* **L2**: 129ms
* **L1**:  91ms
* **Hm32**: 189ms
* **Hm64**: 128ms

