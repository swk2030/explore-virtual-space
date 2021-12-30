#include <InputStream.h>

NetworkStream::NetworkStream(std::string IP, std::string PORT) {
    client = new Client(IP, PORT);
    streamer = new Streamer(client->getServerSock(), client->getServerAddr());
    convertor = new Fmt_Convertor(1024, 4096, 1);
    if (!convertor) {
        std::cerr << __FILE__ << __LINE__ << std::endl;
        exit(1);
    }
    device = new Device(0);
    if (!device) {
        std::cerr << __FILE__ << __LINE__ << std::endl;
        exit(1);
    }
    
}

NetworkStream::~NetworkStream() {
    client->cleanNetwork();
    delete client;
    delete streamer;
    delete convertor;
    delete device;
}

void NetworkStream::recvData(Packet& packet) {
    size_t szBitStream = streamer->recvData();
    uint8_t* bitstream = streamer->getBuffer();
    bd.ptr = bitstream;
    bd.size = szBitStream;
    bd.origPtr = bitstream;
    bd.fullSize = szBitStream;
}
void NetworkStream::decode(Packet& packet) {
    AVIOContext *avio_ctx = NULL;
    bd = { 0,0 };

    uint8_t* avio_ctx_buffer = NULL;
    size_t avio_ctx_buffer_size = 4096;

    int ret = 0;
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
    NvDecoder dec(device->getCuContext(), demuxer.GetWidth(), demuxer.GetHeight(), false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    int nFrame = 0;
    uint8_t* pVideo = NULL;
    int nVideoBytes = 0;
    uint8_t** ppFrame;
    int nFrameReturned = 0;
    size_t next_chunk_begin = *(packet.progress);
    do
    {
        demuxer.Demux(&pVideo, &nVideoBytes);
        dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();
        nFrame += nFrameReturned;
        for (int i = 0; i < nFrameReturned; i++) {
            convert(ppFrame[i], packet.buf + next_chunk_begin);
            *(packet.progress) += packet.chunk_size;
            next_chunk_begin += packet.chunk_size;
        }
    } while (nVideoBytes);
    ZeroMemory(bd.origPtr, bd.fullSize);
}
void NetworkStream::requestData(request &req) {
    SOCKET sock = client->getServerSock();
    send(sock, req.c_str(), req.length(), 0);
}