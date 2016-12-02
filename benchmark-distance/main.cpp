#include <cstdio>
#include <cassert>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <random>
#include "platformMalloc.h"

#if defined _M_X64 || (defined _M_IX86_FP && _M_IX86_FP >= 2)
#include <intrin.h>
#define USE_SSE 1
#elif ((defined WIN32 || defined _WIN32) && defined(_M_ARM)) || defined(__ARM_NEON__) || (defined (__ARM_NEON) && defined(__aarch64__))
#include <arm_neon.h>
#define USE_SSE 0
#else
#error // use either SSE or NEON
#endif

// Parameter D (dimension of a vector with uchar elements)
// has to be set in multiples of ALIGN (i.e., 16).
// This benchmark treats the length of a binary vector as 8D.

const int ALIGN=16; // alignment step for SIMD
const unsigned D=128; // dimension of a vector with uchar elements
const unsigned N=1024*1024*4; // # of dictionary vectors

inline std::string to_binary(unsigned char b)
{
	std::string str("........");
	for(unsigned i=0;i<8;++i)
		if(b&(1<<i))
			str[8-1-i]='#';
	return str;
}

inline void print_vectors(unsigned size,const unsigned char* vecs)
{
	for(unsigned i=0;i<size;++i)
	{
		printf("%2d:  ",i);
		for(unsigned d=0;d<D;++d)
			printf("%3d ",vecs[i*D+d]);
			//printf("%02X ",vecs[i*D+d]);
			//printf("%3s ",to_binary(vecs[i*D+d]).c_str());
		printf("\n");
	}
}

inline int dist_l2(unsigned char* p,unsigned char* q)
{
	int result=0;
	for(unsigned d=0;d<D;++d)
		result+=(q[d]-p[d])*(q[d]-p[d]);
	return result;
}
#if USE_SSE 
inline int dist_l2_simd(unsigned char* p,unsigned char* q)
{
	__m128i t=_mm_setzero_si128();
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		__m128i pm=_mm_load_si128((__m128i*)(p+d));
		__m128i qm=_mm_load_si128((__m128i*)(q+d));
		__m128i sublo=_mm_sub_epi16(
			_mm_unpacklo_epi8(pm,_mm_setzero_si128()),
			_mm_unpacklo_epi8(qm,_mm_setzero_si128())
		);
		t=_mm_add_epi32(t,
			_mm_madd_epi16(sublo,sublo)
		);
		__m128i subhi=_mm_sub_epi16(
			_mm_unpackhi_epi8(pm,_mm_setzero_si128()),
			_mm_unpackhi_epi8(qm,_mm_setzero_si128())
		);
		t=_mm_add_epi32(t,
			_mm_madd_epi16(subhi,subhi)
		);
	}
	return t.m128i_i32[0]+t.m128i_i32[1]+t.m128i_i32[2]+t.m128i_i32[3];
}
#else
inline int dist_l2_simd(unsigned char* p,unsigned char* q)
{
	uint32x4_t t = vdupq_n_u32((unsigned)0);
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		uint8x16_t pm = vld1q_u8(p+d);
		uint8x16_t qm = vld1q_u8(q+d);
        uint16x8_t plo = vmovl_u8(vget_low_u8(pm));
        uint16x8_t qlo = vmovl_u8(vget_low_u8(qm));
        uint16x8_t sublo = vabdq_u16(plo, qlo);
        sublo = vmulq_u16(sublo, sublo);
        t = vaddq_u32(vmovl_u16(vget_low_u16(sublo)), t);
        t = vaddq_u32(vmovl_u16(vget_high_u16(sublo)), t);

        plo = vmovl_u8(vget_high_u8(pm));
        qlo = vmovl_u8(vget_high_u8(qm));
        sublo = vabdq_u16(plo, qlo);
        sublo = vmulq_u16(sublo, sublo);
        t = vaddq_u32(vmovl_u16(vget_low_u16(sublo)), t);
        t = vaddq_u32(vmovl_u16(vget_high_u16(sublo)), t);
    }
    return vgetq_lane_u32(t, 0) + vgetq_lane_u32(t, 1) + vgetq_lane_u32(t, 2) + vgetq_lane_u32(t, 3);
}
#endif

inline int dist_l1(unsigned char* p,unsigned char* q)
{
	int result=0;
	for(unsigned d=0;d<D;++d)
		result+=std::abs(q[d]-p[d]);
	return result;
}
#if USE_SSE
inline int dist_l1_simd(unsigned char* p,unsigned char* q)
{
	__m128i t=_mm_setzero_si128();
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		t=_mm_add_epi32(t,
			_mm_sad_epu8(
				_mm_load_si128((__m128i*)(p+d)),
				_mm_load_si128((__m128i*)(q+d))
			)
		);
	}
	return int(t.m128i_i64[0]+t.m128i_i64[1]);
}
#else
inline int dist_l1_simd(unsigned char* p,unsigned char* q)
{
    uint32x4_t t = vdupq_n_u32(0);
	for(unsigned d=0;d<D;d+=ALIGN)
	{
        t = vaddq_u32(t, vpaddlq_u16(vpaddlq_u8(vabdq_u8(vld1q_u8(p+d), vld1q_u8(q+d)))));
	}
    return vgetq_lane_u32(t, 0) + vgetq_lane_u32(t, 1) + vgetq_lane_u32(t, 2) + vgetq_lane_u32(t, 3);
}
#endif

//bit count by the D&C algorithm
inline int popcount32(unsigned int x)
{
	x=((x&0xAAAAAAAA)>> 1)+(x&0x55555555);
	x=((x&0xCCCCCCCC)>> 2)+(x&0x33333333);
	x=((x&0xF0F0F0F0)>> 4)+(x&0x0F0F0F0F);
	x=((x&0xFF00FF00)>> 8)+(x&0x00FF00FF);
	x=((x&0xFFFF0000)>>16)+(x&0x0000FFFF);
	return int(x);
}
inline int popcount64(unsigned long long x)
{
	x=((x&0xAAAAAAAAAAAAAAAAULL)>> 1)+(x&0x5555555555555555ULL);
	x=((x&0xCCCCCCCCCCCCCCCCULL)>> 2)+(x&0x3333333333333333ULL);
	x=((x&0xF0F0F0F0F0F0F0F0ULL)>> 4)+(x&0x0F0F0F0F0F0F0F0FULL);
	x=((x&0xFF00FF00FF00FF00ULL)>> 8)+(x&0x00FF00FF00FF00FFULL);
	x=((x&0xFFFF0000FFFF0000ULL)>>16)+(x&0x0000FFFF0000FFFFULL);
	x=((x&0xFFFFFFFF00000000ULL)>>32)+(x&0x00000000FFFFFFFFULL);
	return int(x);
}

inline int dist_hamming32(unsigned char* p,unsigned char* q)
{
	int result=0;
	unsigned int* p32=reinterpret_cast<unsigned int*>(p);
	unsigned int* q32=reinterpret_cast<unsigned int*>(q);
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		result+=popcount32((*p32++)^(*q32++));
		result+=popcount32((*p32++)^(*q32++));
		result+=popcount32((*p32++)^(*q32++));
		result+=popcount32((*p32++)^(*q32++));
	}
	return int(result);
}

#if USE_SSE
inline int dist_hamming32_simd(unsigned char* p,unsigned char* q)
{
	int result=0;
	unsigned int* p32=reinterpret_cast<unsigned int*>(p);
	unsigned int* q32=reinterpret_cast<unsigned int*>(q);
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		// this might be faster than using _mm_xor_si128().
		result+=_mm_popcnt_u32((*p32++)^(*q32++));
		result+=_mm_popcnt_u32((*p32++)^(*q32++));
		result+=_mm_popcnt_u32((*p32++)^(*q32++));
		result+=_mm_popcnt_u32((*p32++)^(*q32++));
	}
	return int(result);
}
#endif

inline int dist_hamming64(unsigned char* p,unsigned char* q)
{
	int result=0;
	unsigned long long* p64=reinterpret_cast<unsigned long long*>(p);
	unsigned long long* q64=reinterpret_cast<unsigned long long*>(q);
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		result+=popcount64((*p64++)^(*q64++));
		result+=popcount64((*p64++)^(*q64++));
	}
	return int(result);
}
#if USE_SSE
#if defined (_M_X64)
inline int dist_hamming64_simd(unsigned char* p,unsigned char* q)
{
	long long result=0;
	unsigned long long* p64=reinterpret_cast<unsigned long long*>(p);
	unsigned long long* q64=reinterpret_cast<unsigned long long*>(q);
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		// this is probably faster than using _mm_xor_si128().
		result += _mm_popcnt_u64((*p64++)^(*q64++));
		result += _mm_popcnt_u64((*p64++)^(*q64++));
	}
	return int(result);
}
#endif
#else
inline int dist_hamming64_simd(unsigned char* p,unsigned char* q)
{
	uint16x4_t result = vdup_n_u16(0);
	for(unsigned d=0;d<D;d+=ALIGN)
	{
		uint8x16_t t = vcntq_u8(veorq_u8(vld1q_u8(p+d), vld1q_u8(q+d)));
        uint8x8_t t0 = vpadd_u8(vget_low_u8(t), vget_high_u8(t)); // 16 -> 8
        result = vadd_u16(result, vpaddl_u8(t0)); // 8 -> 4
	}
    return int(vget_lane_u16(result, 0) + vget_lane_u16(result, 1) + vget_lane_u16(result, 2) + vget_lane_u16(result, 3));
}
#endif

inline std::pair<int,int> search(unsigned char* dict,unsigned char* query)
{
	int best_n=-1;
	int best_d=std::numeric_limits<int>::max();
	for(unsigned n=0;n<N;++n)
	{
		// uncomment one of them as you like!
		int d=dist_l2(&dict[n*D],query);
//		int d=dist_l1(&dict[n*D],query);
//		int d=dist_hamming32(&dict[n*D],query);
//		int d=dist_hamming64(&dict[n*D],query);
		if(best_d<d)
			continue;
		best_n=n;
		best_d=d;
	}
	return std::make_pair(best_d,best_n);
}

inline std::pair<int,int> search_sse(unsigned char* dict,unsigned char* query)
{
	int best_n=-1;
	int best_d=std::numeric_limits<int>::max();
	for(unsigned n=0;n<N;++n)
	{
		// uncomment one of them as you like!
		int d=dist_l2_simd(&dict[n*D],query);
//		int d=dist_l1_simd(&dict[n*D],query);
//		int d=dist_hamming32_simd(&dict[n*D],query);
//		int d=dist_hamming64_simd(&dict[n*D],query);
		if(best_d<d)
			continue;
		best_n=n;
		best_d=d;
	}
	return std::make_pair(best_d,best_n);
}

int main()
{
	assert(D%ALIGN==0);

	printf("Dimension of a vector (D): %d\n",D);
	printf("# of dictionary vectors (N): %d\n",N);
	printf("-----------------------------------------\n");

	std::mt19937 rng;
	std::uniform_real_distribution<float> dist(0.,255.);
	
	unsigned char* dict=reinterpret_cast<unsigned char*>(alignedMalloc(N*D,ALIGN));
	unsigned char* query=reinterpret_cast<unsigned char*>(alignedMalloc(D,ALIGN));
	
	// generate dictionary and query vectors randomly
	printf("[vector generation]\n");
	for(unsigned n=0;n<N;++n)
	{
		if((n+1)%1024==0)
			printf("  Progress... (%5dk / %5dk)\r",(n+1)/1024,N/1024);
		for(unsigned d=0;d<D;++d)
			dict[n*D+d]=(unsigned char)dist(rng);
	}
//	printf("\n");
//	for(unsigned d=0;d<D;++d)
//		query[d]=(unsigned char)dist(rng);

	//// print vectors
	//printf("[dictionary vectors]\n");
	//print_vectors(N,dict);
	//printf("[query vectors]\n");
	//print_vectors(1,query);

	printf("[full nearest neighbor search w/o SSE]\n");
	clock_t tickA0=clock();
	std::pair<int,int> resultA=search(dict,query);
	clock_t tickA1=clock();
	double timeA=double(tickA1-tickA0)*1000.0/CLOCKS_PER_SEC;
	printf("  Nearest neighbor:  %d (distance=%d)\n",resultA.second,resultA.first);
	printf("  Search time:  %6.0f [ms]\n",timeA);
	
	printf("[full nearest neighbor search w/ SSE]\n");
	clock_t tickB0=clock();
	std::pair<int,int> resultB=search_sse(dict,query);
	clock_t tickB1=clock();
	double timeB=double(tickB1-tickB0)*1000.0/CLOCKS_PER_SEC;
	printf("  Nearest neighbor:  %d (distance=%d)\n",resultB.second,resultB.first);
	printf("  Search time:  %6.0f [ms]\n",timeB);

	alignedFree(dict);
	alignedFree(query);
	return 0;
}
