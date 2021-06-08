from __future__ import print_function, division
import argparse

parser = argparse.ArgumentParser(description="Generate some tracks")

parser.add_argument("--trk", type=str, default=1, help=".txt file containing track info")
parser.add_argument("--truth_trk", type=str, default=1, help=".txt file containing truth track info")
parser.add_argument("--vrtx", type=str, default=1, help=".txt file containing vertex info")
parser.add_argument("--truth_vrtx", type=str, default=1, help=".txt file containing truth vertex info")

parser.add_argument("name", help="The name of the file to produce: .../trks_name.root")

args = parser.parse_args()

import os
import sys
import time
import numpy as np
import ROOT
import pandas as pd
import awkward as ak

from writer import Writer


def run(lname, trk, truth_trk, vrtx, truth_vrtx):
    #name = "/data/schreihf/PvFinder/pv_{}.root".format(lname)
    print("Converting tracks to ROOT file at ", lname)
    
    
    #read in track info
    track_info = pd.read_csv(trk, header = None, delim_whitespace=True,
                             names = ("name","lumi","evt","num","good","pt","unknown1","cottheta","phi","d0","z0"),
                             usecols = [i for i in range(11)],
                             dtype={"good": bool})
    #delete name (don't need, same for all entries) and set 2-level index
    del track_info["name"]
    track_info.set_index(["evt", "num"], inplace=True)
    track_info = track_info[track_info.good]
    
    
    # read in truth track info
    truth_track_info = pd.read_csv(truth_trk, header = None, delim_whitespace=True)
    truth_track_info.columns = ["name","lumi","evt","num","charge","px","py","pz","vertexX","vertexY","vertexZ"]
    del truth_track_info["name"]
    truth_track_info.set_index(["evt", "num"], inplace=True)
    
    
    # read in vertex info
    wf = open(vrtx, 'r')
    lines = wf.readlines()

    trk_pts = []
    counts = []
    skipped = []
    for j, line in enumerate(lines):
        try:
            current = line.split(maxsplit = 11)[11]
            current_pts = np.ndarray.tolist(np.fromstring(current, dtype=float, sep=' '))
            counts.append(len(current_pts))
            trk_pts = trk_pts + current_pts
        except Exception: 
            skipped.append(j)
            continue
        
    pts_ak = ak.JaggedArray.fromcounts(counts,trk_pts)
    
    if len(skipped) == 0:
        skip = None
    else: skip = skipped
    
    cols = ["name","lumi","evt","num","x","y","z","dx","dy","dz","n_trks"]
    vertex_info = pd.read_csv(vrtx, header = None, delim_whitespace=True,
                              usecols = [i for i in range(11)], skiprows=skip)
    vertex_info.columns = cols

    del vertex_info["name"]
    vertex_info.set_index(["evt", "num"], inplace=True)
    vertex_info["trk_pts"] = pts_ak
    
    wf.close()
    
    
    # read in truth vertex info
    wf = open(truth_vrtx, 'r')
    lines = wf.readlines()

    trk_nums = []
    counts = []
    skipped = []
    for j, line in enumerate(lines):
        try:
            current = line.split(maxsplit = 8)[8]
            nums = np.ndarray.tolist(np.fromstring(current, dtype=int, sep=' '))
            counts.append(len(nums))
            trk_nums = trk_nums + nums
        except Exception: 
            skipped.append(j)
            continue

    nums_ak = ak.JaggedArray.fromcounts(counts,trk_nums)
        
    if len(skipped) == 0:
        skip = None
    else: skip = skipped
        
    cols = ["name","lumi","evt","num","x","y","z","n_trks"]
    truth_vertex_info = pd.read_csv('truch_vertex_info_v3.txt',
                                    header = None,
                                    delim_whitespace=True, 
                                    usecols = [i for i in range(8)],
                                    engine = 'python',
                                    skiprows = skip)

    truth_vertex_info.columns = cols

    del truth_vertex_info["name"]
    truth_vertex_info.set_index(["evt", "num"], inplace=True)
    truth_vertex_info["trk_nums"] = nums_ak
    
    wf.close()
    
    
    
    
    
    # Create the output TFile and TTree.
    tfile = ROOT.TFile(name, "RECREATE")
    ttree = ROOT.TTree("data", "")

    # Create the writer handler, add branches, and initialize.
    writer = Writer(ttree) 
    writer.add("pvr_x", "pvr_y", "pvr_z", "svr_x", "svr_y", "svr_z", "svr_pvr",
              "hit_x", "hit_y", "hit_z", "hit_prt", "prt_pid", "prt_px", "prt_py", "prt_pz",
              "prt_hits", "prt_pvr", "ntrks_prompt")
    writer.add("pvr_y")
    writer.add("pvr_z")
    writer.add("svr_x")
    writer.add("svr_y")
    writer.add("svr_z")
    writer.add("svr_pvr")
    writer.add("hit_x")
    writer.add("hit_y")
    writer.add("hit_z")
    writer.add("hit_prt")
    writer.add("prt_pid")
    writer.add("prt_px")
    writer.add("prt_py")
    writer.add("prt_pz")
    writer.add("prt_e")
    writer.add("prt_x")
    writer.add("prt_y")
    writer.add("prt_z")
    writer.add("prt_hits")
    writer.add("prt_pvr")
    writer.add("ntrks_prompt")
    
    
    # only use events common to both dataframes
    evts_trk = set(track_info.index.get_level_values(0).tolist())
    evts_vrtx = set(truth_vertex_info.index.get_level_values(0).tolist())
    evts_invalid = evts_trk ^ evts_vrtx
    track_info = track_info.drop(index = sorted(list(evts_invalid & evts_trk)), level = 0)
    truth_vertex_info = truth_vertex_info.drop(index = sorted(list(evts_invalid & evts_vrtx)), level = 0)

    
    # Fill the events.

    ttime = 0
    for vrtx, trk in zip(truth_vertex_info.groupby("evt"), track_info.groupby("evt")):
    
        start = time.time()
    
    
        # truth matching
        prt_pvr = []
        trk_nlist = trk[1].index.get_level_values("num").tolist()
        trk_nlists = vrtx[1]["trk_nums"].tolist()
        vrtx_nlist = vrtx[1].index.get_level_values("num").tolist()
    
        nfound = 0
    
        for i in range(len(trk_nlist)):
            found = False
            for j in range(len(trk_nlists)):
                if trk_nlist[i] in trk_nlists[j]:
                    writer["prt_pvr"].append(vrtx_nlist[j])
                    nfound+=1
                    found = True
                    break
            if not found:
                writer["prt_pvr"].append(None)
    
        print('Event Num: ', vrtx[0])
        print('Matches Found: ', nfound)
        print()
    
        #append vertex info
        xlist = vrtx[1]["x"].tolist()
        ylist = vrtx[1]["y"].tolist()
        zlist = vrtx[1]["z"].tolist()
    
        for i in range(len(xlist)):
        
            writer["pvr_x"].append(xlist[i])
            writer["pvr_y"].append(ylist[i])
            writer["pvr_z"].append(zlist[i])
    
        #grab useful values
        d0 = np.array(trk[1]["d0"].tolist())
        phi = np.array(trk[1]["phi"].tolist())
        z0 = np.array(trk[1]["z0"].tolist())
        cottheta = np.array(trk[1]["cottheta"].tolist())
        
        #create location vector
        loc = np.stack((np.multiply(d0,np.cos(np.add(phi,-np.pi/2))),
                        np.multiply(d0,np.sin(np.add(phi,-np.pi/2))), 
                        z0))
        
        #create direction vector
        direc = np.stack((np.cos(phi), np.sin(phi), cottheta))
    
        #normalize vectors
        loc = np.divide(loc, np.linalg.norm(loc, axis = 1).reshape(3,1))
        direc = np.divide(direc, np.linalg.norm(direc, axis = 1).reshape(3,1))
    
        # append track info
        for i in range(len(loc[0])):
            writer["prt_x"].append(loc[0][i])
            writer["prt_y"].append(loc[1][i])
            writer["prt_z"].append(loc[2][i])
            
            writer["prt_px"].append(direc[0][i])
            writer["prt_py"].append(direc[1][i])
            writer["prt_pz"].append(direc[2][i])
    
        ttree.Fill()
        itime = time.time() - start
        ttime += itime
        writer.clear()

    ttree.Print()
    ttree.Write(ttree.GetName(), ROOT.TObject.kOverwrite)
    tfile.Close()
    
    return ttime


if __name__ == "__main__":
    
    stime = run(args.name, args.trk, args.truth_trk, args.vrtx, args.truth_vrtx)
    
    print("Wrote ROOT file in in {:.5} s".format(stime))
