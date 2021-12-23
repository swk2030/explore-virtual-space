#include "Conversion.h"

Fmt_Convertor::Fmt_Convertor(int W, int H, int frameCnt) {
	this->frameCnt = frameCnt;
	size.width = W;
	size.height = H;
	oSizeROI.width = W;
	oSizeROI.height = H;

	widthByte = size.width * 1.5;
	height = size.height;
#if true
	cudaMalloc(&rgb, W * H * 3);
	cudaMalloc(&yuv, W * H * 1.5);
#else
#endif // false
}
Fmt_Convertor::~Fmt_Convertor() {
	cudaFree(yuv);
	cudaFree(rgb);
}
void Fmt_Convertor::Frame_NV12ToRGB(Npp8u* nv12_data, Npp8u* rgb_data) {
	cudaMemcpy(yuv, nv12_data, size.width * size.height * 1.5, cudaMemcpyHostToDevice);
	const Npp8u* pSrc[2] = { yuv, yuv + size.width * size.height };
	NppStatus stat = nppiNV12ToRGB_8u_P2C3R(pSrc, size.width, rgb, size.width * 3 * sizeof(Npp8u), oSizeROI);
	if (stat != NPP_SUCCESS) { std::cout << "Output NPP error" << std::endl; return; }
	cudaMemcpy(rgb_data, rgb, size.width * size.height * 3, cudaMemcpyDeviceToHost);
}


void Fmt_Convertor::Video_NV12ToRGB(Npp8u* nv12_datas, Npp8u* rgb_datas) {
	float yuv_factor = 1.5;
	for (int i = 0; i < frameCnt; i++) {
		Frame_NV12ToRGB(nv12_datas, rgb_datas);
		nv12_datas += size.width * size.height + (size.width*size.height/2);
		rgb_datas += size.width * size.height * 3;
	}
}

void Fmt_Convertor::Video_NV12ToRGB(Npp8u** nv12_datas, Npp8u* rgb_datas) {
	float yuv_factor = 1.5;
	for (int i = 0; i < frameCnt; i++) {
		Frame_NV12ToRGB(nv12_datas[i], rgb_datas);
		//nv12_datas += size.width * size.height + (size.width * size.height / 2);
		rgb_datas += size.width * size.height * 3;
	}
}