#pragma once
#include <cuda.h>
#include "NvDecoder/NvDecoder.h"        
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"
#include "Device.h"
#include "Conversion.h"

#define UNROLLING_CODE 0
#define FFMPEG_CUSTOM 0

class NvDec {
private:
    NvDecoder* decoder;
    Fmt_Convertor* convertor;
#if FFMPEG_CUSTOM
    FFmpegDemuxer* demuxer;
    uint8_t* custom_buffer;
#endif // FFMPEG_CUSTOME

    int width;
    int height;
    int frameCnt;
    Device* device;
    Npp8u* buffer;
public:
    NvDec(int iGpu, int width, int height, int frameCnt);
    ~NvDec();
    uint8_t** Decode_custom(unsigned char* iBuffer, int size, bool bOutPlanar);
    uint8_t** Decode(unsigned char* iBuffer, int size, bool bOutPlanar);
    uint8_t* Decode_(unsigned char* iBuffer, int size, bool bOutPlanar);
    void Decode_Directly(unsigned char* iBuffer, int size, bool bOutPlanar);
    void Decode(unsigned char* iBuffer, unsigned char** oBuffer, int size, bool bOutPlanar);
    void clearVector() { decoder->init_dPtr(); }
    void pushBackVector(uint8_t* ptr) { decoder->push_dPtr(ptr); }
    void writeData(const char path[], unsigned char** data);
    void ConvertToPlanar(uint8_t* pHostFrame, uint8_t* pRGBHostFrame);
};