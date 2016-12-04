# 実行ファイル名とソース指定
add_executable(benchmark-distance main.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
	set(ARM 1)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfp16-format=ieee -mfpu=neon-fp16 -flax-vector-conversions -std=c++11")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64)")
	set(AARCH64 1)
else ()
	set(X86 1)
endif()

set_property(TARGET benchmark-distance PROPERTY CXX_STANDARD 11)
