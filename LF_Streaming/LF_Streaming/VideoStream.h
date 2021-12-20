#pragma once
#include <iostream>
#include <fstream>
extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
}

struct buffer_in_data {
    //ptr�� size�� ���� stream�� pointer�� ���� ũ��
    //origPtr�� fullSize�� seek�Լ����� pointer ������ �� ���Ǵ� ����
    uint8_t* ptr;
    uint8_t* origPtr;  
    size_t size;
    size_t fullSize;
};


// buf�� �ִ� �����͸� opaque�� stream���� �ű�� �۾� ����
static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
    struct buffer_in_data* bd = (struct buffer_in_data*)opaque;
    buf_size = FFMIN(buf_size, bd->size);
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;
    return buf_size;
}


/*
opaque�� stream���� pointer�� �����ϴ� �Լ�
file stream�� seek�Լ��� ������ ���
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