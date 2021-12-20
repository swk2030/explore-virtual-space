#include "NvDec.h"
simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

NvDec::NvDec(int iGpu, int width, int height, int frameCnt) {
    Rect cropRect = {};
    Dim resizeDim = {};
    this->width = width;
    this->height = height;
    this->frameCnt = frameCnt;
    convertor = new Fmt_Convertor(width, height, frameCnt);
    device = new Device(iGpu);
    decoder = new NvDecoder(device->getCuContext(), width, height, false, FFmpeg2NvCodecId(AV_CODEC_ID_H264), NULL, false, false, &cropRect, &resizeDim);
    cudaMallocHost(&buffer, 10 * 1024 * 256 * 3);

#if FFMPEG_CUSTOM
    demuxer = new FFmpegDemuxer(0);
#endif
    //av_register_all();
    //avformat_network_init();
}

NvDec::~NvDec() {
    if (decoder != NULL)	delete decoder;
    if (device != NULL)	delete device;
    if (convertor != NULL) delete convertor;
    cudaFreeHost(buffer);
#if FFMPEG_CUSTOM
    delete demuxer;
#endif // FFMPEG_CUSTOM

}

void NvDec::writeData(const char* path, unsigned char** data) {
    std::ofstream fpOut(path, std::ios::out | std::ios::binary);
    if (!fpOut)
    {
        std::ostringstream err;
        err << "Unable to open output file: " << path << std::endl;
        throw std::invalid_argument(err.str());
    }

    for (int i = 0; i < 10; i++) {
        fpOut.write(reinterpret_cast<char*>(data[i]), width*height*1.5);
    }
}


void NvDec::ConvertToPlanar(uint8_t* pHostFrame, uint8_t* pRGBHostFrame) {
    convertor->Frame_NV12ToRGB(pHostFrame, pRGBHostFrame);
}

uint8_t** NvDec::Decode(unsigned char* iBuffer, int iSize, bool bOutPlanar) {
    Npp8u** data = new Npp8u *[frameCnt];
    float bpp = bOutPlanar ? 3 : 1.5;
    for (int i = 0; i < frameCnt; i++) {
        data[i] = new Npp8u[width * height * bpp];
    }
    AVIOContext* ioCtx = avio_alloc_context(
        iBuffer,    // Buffer
        iSize,  // Buffer size
        0,                   // Buffer is only readable - set to 1 for read/write
        NULL,      // User (your) specified data
        NULL,      // Function - Reading Packets (see example)
        NULL,                   // Function - Write Packets
        NULL       // Function - Seek to position in stream (see example)
    );
    if (ioCtx == nullptr) {
        // out of memory
    }
    FFmpegDemuxer demux(ioCtx);

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, **ppFrame;
    int count = 0;

    do {
        demux.Demux(&pVideo, &nVideoBytes);
        decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

        if (!nFrame && nFrameReturned)
            //LOG(INFO) << decoder->GetVideoInfo();

            for (int i = 0; i < nFrameReturned; i++) {
                if (bOutPlanar) {
                    ConvertToPlanar(ppFrame[i], data[count++]);
                }
                else {
                    memcpy(data[count++], ppFrame[i], decoder->GetFrameSize());
                }
            }
        nFrame += nFrameReturned;

    } while (nVideoBytes);

    //std::cout << "Decoding was done..." << std::endl;
    return data;
}

uint8_t** NvDec::Decode_custom(unsigned char* iBuffer, int iSize, bool bOutPlanar) {
    Npp8u** data = new Npp8u *[frameCnt];
    float bpp = bOutPlanar ? 3 : 1.5;
    for (int i = 0; i < frameCnt; i++) {
        data[i] = new Npp8u[width * height * bpp];
    }
    AVIOContext* ioCtx = avio_alloc_context(
        iBuffer,    // Buffer
        iSize,  // Buffer size
        0,                   // Buffer is only readable - set to 1 for read/write
        NULL,      // User (your) specified data
        NULL,      // Function - Reading Packets (see example)
        NULL,                   // Function - Write Packets
        NULL       // Function - Seek to position in stream (see example)
    );
    if (ioCtx == nullptr) {
        // out of memory
    }
#if FFMPEG_CUSTOM
    demuxer->Before_Demux(ioCtx);
#else
    FFmpegDemuxer demux(ioCtx);
#endif // FFMPEG_CUSTOM

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, ** ppFrame;
    int count = 0;

    do {
#if FFMPEG_CUSTOM
        demuxer->Demux(&pVideo, &nVideoBytes);
#else
        demux.Demux(&pVideo, &nVideoBytes);
#endif // FFMPEG_CUSTOM

        decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

        if (!nFrame && nFrameReturned);
        //LOG(INFO) << decoder->GetVideoInfo();

        for (int i = 0; i < nFrameReturned; i++) {
            if (bOutPlanar) {
                ConvertToPlanar(ppFrame[i], data[count++]);
            }
            else {
                memcpy(data[count++], ppFrame[i], decoder->GetFrameSize());
            }
        }
        nFrame += nFrameReturned;

    } while (nVideoBytes);

    //std::cout << "Decoding was done..." << std::endl;
    return data;
}

uint8_t* NvDec::Decode_(unsigned char* iBuffer, int iSize, bool bOutPlanar) {
#if true
    AVIOContext* ioCtx = avio_alloc_context(
        iBuffer,    // Buffer
        iSize,  // Buffer size
        0,                   // Buffer is only readable - set to 1 for read/write
        NULL,      // User (your) specified data
        NULL,      // Function - Reading Packets (see example)
        NULL,                   // Function - Write Packets
        NULL       // Function - Seek to position in stream (see example)
    );

    if (ioCtx == nullptr) {
        // out of memory
    }
    FFmpegDemuxer demux(ioCtx);
#else 
    ioCtx->buffer = iBuffer;
    ioCtx->buffer_size = iSize;
    FFmpegDemuxer demux(ioCtx);
#endif // false

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, ** ppFrame;
    int count = 0;

    do {
        demux.Demux(&pVideo, &nVideoBytes);
        decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        if (!nFrame && nFrameReturned) {
            LOG(INFO) << decoder->GetVideoInfo();
        }
        for (int i = 0; i < nFrameReturned; i++) {
            if (bOutPlanar) {
                ConvertToPlanar(ppFrame[i], buffer);
                buffer += 1024 * 256 * 3;
            }
        }
        nFrame += nFrameReturned;

    } while (nVideoBytes);
    buffer -= 1024 * 256 * 3 * 10;
    //std::cout << "Decoding was done..." << std::endl;
    return buffer;
}

void NvDec::Decode_Directly(unsigned char* iBuffer, int iSize, bool bOutPlanar) {
    AVIOContext* ioCtx = avio_alloc_context(
        iBuffer,    // Buffer
        iSize,  // Buffer size
        0,                   // Buffer is only readable - set to 1 for read/write
        NULL,      // User (your) specified data
        NULL,      // Function - Reading Packets (see example)
        NULL,                   // Function - Write Packets
        NULL       // Function - Seek to position in stream (see example)
    );

    if (ioCtx == nullptr) {
        // out of memory
    }
#if FFMPEG_CUSTOM
    demuxer->Before_Demux(ioCtx);
#else
    FFmpegDemuxer demux(ioCtx);
#endif // FFMPEG_CUSTOM


    //auto start = std::chrono::high_resolution_clock::now();
    //auto done = std::chrono::high_resolution_clock::now();
    //std::cout << "File Read time : " << std::chrono::duration_cast<std::chrono::milliseconds>(done - start).count() << "ms " << std::endl;
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, ** ppFrame;
    int count = 0;
    //auto start = std::chrono::high_resolution_clock::now();
    //
    //auto done = std::chrono::high_resolution_clock::now();
    //std::cout << "Execution time : " << std::chrono::duration_cast<std::chrono::microseconds>(done - start).count() << "us " << std::endl;

#if UNROLLING_CODE
    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

    demux.Demux(&pVideo, &nVideoBytes);
    decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
#else	
    LOG(INFO) << decoder->GetVideoInfo();

    do {
#if FFMPEG_CUSTOM
        demuxer->Demux(&pVideo, &nVideoBytes);
#else
        demux.Demux(&pVideo, &nVideoBytes);
#endif // FFMPEG_CUSTOM

        decoder->Decode_Directly(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

        if (!nFrame && nFrameReturned) {
            //LOG(INFO) << decoder->GetVideoInfo();
        }
        nFrame += nFrameReturned;
        //count++;
        //printf("count : %d\n", count);
    } while (nVideoBytes);
#endif // Unrolling_Code

    //std::cout << "Decoding was done..." << std::endl;
}
void NvDec::Decode(unsigned char* iBuffer, unsigned char** oBufferFrame, int iSize, bool bOutPlanar) {
    AVIOContext* ioCtx = avio_alloc_context(
        iBuffer,    // Buffer
        iSize,  // Buffer size
        0,                   // Buffer is only readable - set to 1 for read/write
        NULL,      // User (your) specified data
        NULL,      // Function - Reading Packets (see example)
        NULL,                   // Function - Write Packets
        NULL       // Function - Seek to position in stream (see example)
    );

    if (ioCtx == nullptr) {
        // out of memory
    }
    FFmpegDemuxer demux(ioCtx);
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, ** ppFrame;
    int count = 0;

    do {
        demux.Demux(&pVideo, &nVideoBytes);
        decoder->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << decoder->GetVideoInfo();
        for (int i = 0; i < nFrameReturned; i++) {
            if (bOutPlanar) {
                ConvertToPlanar(ppFrame[i], oBufferFrame[count++]);
            }
            else {
                memcpy(oBufferFrame[count++], ppFrame[i], decoder->GetFrameSize());
            }
        }
        nFrame += nFrameReturned;

    } while (nVideoBytes);

    std::cout << "Decoding was done..." << std::endl;
}