#pragma once
#include <iostream>
#include <fstream>
#include "NvDecoder/NvDecoder.h"
#include "Utils/NvCodecUtils.h"
#include "AppDecUtils.h"
#include "Utils/FFmpegDemuxer.h"
#include "Conversion.h"
extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
}

Fmt_Convertor* convertor;
struct buffer_in_data {
    //ptr과 size는 현재 stream의 pointer와 읽은 크기
    //origPtr과 fullSize는 seek함수에서 pointer 조정할 때 사용되는 변수
    uint8_t* ptr;
    uint8_t* origPtr;  
    size_t size;
    size_t fullSize;
};


// buf에 있는 데이터를 opaque의 stream으로 옮기는 작업 진행
static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
    struct buffer_in_data* bd = (struct buffer_in_data*)opaque;
    buf_size = FFMIN(buf_size, bd->size);
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;
    return buf_size;
}


/*
opaque의 stream에서 pointer를 조정하는 함수
file stream의 seek함수와 동일한 기능
*/
static int64_t seek_packet(void* opaque, std::int64_t where, int whence) {
    auto me = reinterpret_cast<buffer_in_data*>(opaque);
    switch (whence) {
    case AVSEEK_SIZE:
        return me->fullSize;
        break;
    case SEEK_SET:
        if (me->fullSize > where) {
            me->ptr = me->origPtr + where;
            me->size = me->fullSize - where;
        }
        else {
            return EOF;
        }
        break;
    case SEEK_END:
        if (me->fullSize > where) {
            me->ptr = (me->origPtr + me->fullSize) - where;
            int curPos = me->ptr - me->origPtr;
            me->size = me->fullSize - curPos;
        }
        else {
            return EOF;
        }
        break;
    default:
        /* On error, do nothing, return current position of file. */
        std::cerr << "Could not process buffer seek: "
            << whence << ".\n";
        break;
    }
    return me->ptr - me->origPtr;
}

//@Test function
void file_map(const char *szInFilePath, uint8_t** bufptr, size_t* size) {
    std::ifstream fp;
    fp.open(szInFilePath, std::ios::binary);
    fp.seekg(0, std::ios::end);
    *size = fp.tellg();
    fp.seekg(0, std::ios::beg);

    *bufptr = (uint8_t*)av_malloc(*size);
    fp.read((char*)*bufptr, *size);
    fp.close();
}

//@Test function
void readMp4(CUcontext& cuContext, std::string filename, uint8_t* LF, size_t chunk_size, size_t num_chunk) {
    convertor = new Fmt_Convertor(1024, 4096, 50);
    uint8_t* buffer = NULL;
    size_t size = 0;
    file_map(filename.c_str(), &buffer, &size);
    bool bOutPlanar = false;
    AVIOContext* avio_ctx = NULL;
    struct buffer_in_data bd = { 0, 0 };
    uint8_t* avio_ctx_buffer = NULL;
    size_t avio_ctx_buffer_size = 4096;

    int ret = 0;
    bd.ptr = buffer;
    bd.origPtr = buffer;
    bd.size = size;
    bd.fullSize = size;

    avio_ctx_buffer = (uint8_t*)av_malloc(avio_ctx_buffer_size);
    if (!avio_ctx_buffer) {
        ret = AVERROR(ENOMEM);
        std::cout << "return " << ret << std::endl;
    }
    avio_ctx = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 0
        , &bd, &read_packet, NULL, &seek_packet);
    if (!avio_ctx) {
        ret = AVERROR(ENOMEM);
        std::cout << "return " << ret << std::endl;
    }
    FFmpegDemuxer demuxer(avio_ctx);
    NvDecoder dec(cuContext, demuxer.GetWidth(), demuxer.GetHeight(), false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    int nFrame = 0;
    uint8_t* pVideo = NULL;
    int nVideoBytes = 0;
    uint8_t** ppFrame;
    int nFrameReturned = 0;
    uint8_t** data = new uint8_t *[50];
    float bpp = bOutPlanar ? 3 : 1.5;
    for (int i = 0; i < 50; i++) {
        data[i] = new Npp8u[1024 * 4096* bpp];
    }
    int count = 0;
    do
    {
        demuxer.Demux(&pVideo, &nVideoBytes);
        dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();

        nFrame += nFrameReturned;
        for (int i = 0; i < nFrameReturned; i++) {
            convertor->Frame_NV12ToRGB(ppFrame[i], data[count]);
            memcpy(LF, data[count++], dec.GetFrameSize());
            //fpOut.write(reinterpret_cast<char*>(ppFrame[i]), dec.GetFrameSize());
        }
    } while (nVideoBytes);
    av_free(buffer);
}
