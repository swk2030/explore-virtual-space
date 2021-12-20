#pragma once
#include <cuda.h>
#include "NvDecoder/NvDecoder.h"        
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"

class Device {
private:
	CUdevice device;
	CUcontext cuContext;
public:
	Device(int iGPU) {
		ck(cuInit(0));
		int nGpu = 0;
		ck(cuDeviceGetCount(&nGpu));
		if (iGPU < 0 || iGPU >= nGpu) {
			std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
			return;
		}
		ck(cuDeviceGet(&device, iGPU));
		char szDeviceName[80];
		ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), device));
		cuContext = NULL;
		ck(cuCtxCreate(&cuContext, 0, device));
		std::cout << "GPU initialization was completed..." << std::endl;
	}
	CUcontext getCuContext() { return cuContext; }

};