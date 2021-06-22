#pragma once


#define PVF_XSTR(s) PVF_STR(s)
#define PVF_STR(s) #s

#include <TTree.h>
#include <array>

class DataKernelOut {
    TTree *t;

  public:
    std::array<float, NBINS> zdata{};
    std::array<float, NBINS> xmax{};
    std::array<float, NBINS> ymax{};

    DataKernelOut(TTree *tree) : t(tree) {
        t->Branch("zdata", zdata.data(), "zdata[" PVF_XSTR(NBINS) "]/F");
        t->Branch("xmax", xmax.data(), "xmax[" PVF_XSTR(NBINS) " ]/F");
        t->Branch("ymax", ymax.data(), "ymax[" PVF_XSTR(NBINS) "]/F");
    }

    void clear() {
        zdata.fill(0);
        xmax.fill(0);
        ymax.fill(0);
    }
};

#undef PVF_XSTR
#undef PVF_STR
