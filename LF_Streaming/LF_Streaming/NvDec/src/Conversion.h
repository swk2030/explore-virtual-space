#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <nppi.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <thread>

class Fmt_Convertor {
private:
	int frameCnt;
	Npp8u* rgb;
	Npp8u* yuv;
	NppiSize oSizeROI;
	NppiSize size;
	
	size_t pitch = 0;
	size_t widthByte;
	size_t height;
public:
	Fmt_Convertor(int W, int H, int = 1);
	~Fmt_Convertor();
	void Frame_NV12ToRGB(Npp8u* nv12_data, Npp8u* rgb_data);
	void Video_NV12ToRGB(Npp8u* nv12_data, Npp8u* rgb_data);
	void Video_NV12ToRGB(Npp8u** nv12_data, Npp8u* rgb_data);
};

