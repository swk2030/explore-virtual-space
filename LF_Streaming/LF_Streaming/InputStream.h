#pragma once
#include <iostream>
#include <Device.h>
#include <Conversion.h>
#include <Client.h>
#include <Network.h>
#include <Streamer.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
}
static int read_packet(void* opaque, uint8_t* buf, int buf_size);
static int64_t seek_packet(void* opaque, std::int64_t where, int whence);

struct buffer_in_data {
    uint8_t* ptr;
    uint8_t* origPtr;
    size_t size;
    size_t fullSize;
};

class InputStream {
private:
    buffer_in_data bd;
    Fmt_Convertor *convertor;
    Device* device;

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

public:
    InputStream(int iGpu) {
        convertor = new Fmt_Convertor(1024, 4096, 1);
        if (!convertor) {
            std::cerr<< __FILE__ << __LINE__ << std::endl;
            exit(1);
        }
        device = new Device(0);
        if (!device) {
            std::cerr << __FILE__ << __LINE__ << std::endl;
            exit(1);
        }
    }
    ~InputStream() {
        delete convertor;
        delete device;
    }
    void setBD(buffer_in_data& bd) {
        this->bd = bd;
    }
    buffer_in_data getBD() { return bd; }
    void convert(uint8_t* in, uint8_t* out) {
        convertor->Frame_NV12ToRGB(in, out);
    }
    
    void decode_LF(std::string filename, uint8_t* buf, size_t num_chunk, size_t chunk_size, size_t* progress) {
        uint8_t* bitstream = NULL;
        size_t szBitstream = 0;
        file_map(filename.c_str(), &bitstream, &szBitstream);
        bool bOutPlanar = false;
        AVIOContext* avio_ctx = NULL;
        bd = { 0,0 };

        uint8_t* avio_ctx_buffer = NULL;
        size_t avio_ctx_buffer_size = 4096;
        int ret = 0;
        bd.ptr = bitstream;
        bd.size = szBitstream;
        bd.origPtr = bitstream;
        bd.fullSize = szBitstream;
        avio_ctx_buffer = (uint8_t*)av_malloc(avio_ctx_buffer_size);
        if (!avio_ctx_buffer) {
            ret = AVERROR(ENOMEM);
            std::cout << "return " << ret << std::endl;
        }
        avio_ctx = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 0
            , &bd, &read_packet, NULL, &seek_packet);
        //avio_ctx = avio_alloc_context(iBuffer, iSize, 0, NULL, NULL, NULL, NULL);
        if (!avio_ctx) {
            ret = AVERROR(ENOMEM);
            std::cout << "return " << ret << std::endl;
        }

        FFmpegDemuxer demuxer(avio_ctx);
        NvDecoder dec(device->getCuContext(), demuxer.GetWidth(), demuxer.GetHeight(), false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));

        int nFrame = 0;
        uint8_t* pVideo = NULL;
        int nVideoBytes = 0;
        uint8_t** ppFrame;
        int nFrameReturned = 0;
        size_t next_chunk_begin = *progress;
        do
        {
            demuxer.Demux(&pVideo, &nVideoBytes);
            dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
            if (!nFrame && nFrameReturned)
                LOG(INFO) << dec.GetVideoInfo();
            nFrame += nFrameReturned;
            for (int i = 0; i < nFrameReturned; i++) {
                //*progress += fread(buf + next_chunk_begin, 1, sizeof(uint8_t) * chunk_size, fp);
                convert(ppFrame[i], buf + next_chunk_begin);
                *progress += chunk_size;
                next_chunk_begin += chunk_size;
            }
        } while (nVideoBytes);

        av_free(bitstream);
    }


};
typedef struct Packet {
    uint8_t* buf;
    size_t chunk_size;
    size_t* progress;
};
typedef std::string request;

class NetworkStream {
private:
    buffer_in_data bd;
    Fmt_Convertor *convertor;
    Device *device;
    Streamer *streamer;
    Client *client;

    
    void convert(uint8_t* in, uint8_t* out) {convertor->Frame_NV12ToRGB(in, out);}
public:
    NetworkStream(std::string IP, std::string PORT);
    ~NetworkStream();
    void decode(Packet &packet);
    void recvData();
    void requestData(request &req);
};



static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
    struct buffer_in_data* bd = (struct buffer_in_data*)opaque;
    buf_size = FFMIN(buf_size, bd->size);
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;
    return buf_size;
}

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


/*
@Input : filename
*/
//void decode_LF(){
//
//}