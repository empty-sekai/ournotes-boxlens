"""Bound only this benchmark process to a fixed CPU set; record wall-clock latency."""
import argparse
import ctypes
import hashlib
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from .assets import write
from .engine import Engine


def memory_usage():
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('page_faults',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ['peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        value=Counters();value.cb=ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):raise ctypes.WinError(ctypes.get_last_error())
        return {'rss_bytes':value.rss,'peak_rss_bytes':value.peak_rss}
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peak_rss_bytes':int(peak*(1 if platform.system()=='Darwin' else 1024))}

def bind_cpus(count):
    if hasattr(os,'sched_getaffinity'):
        available=sorted(os.sched_getaffinity(0))
        if len(available)<count:raise RuntimeError('Not enough available logical CPUs')
        chosen=available[:count];os.sched_setaffinity(0,chosen)
        return sorted(os.sched_getaffinity(0))
    if os.name=='nt':
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
        handle=kernel.GetCurrentProcess();process=ctypes.c_size_t();system=ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle,ctypes.byref(process),ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
        available=[i for i in range(8*ctypes.sizeof(process)) if process.value&(1<<i)]
        if len(available)<count:raise RuntimeError('Not enough available logical CPUs')
        mask=sum(1<<i for i in available[:count])
        if not kernel.SetProcessAffinityMask(handle,mask):raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GetProcessAffinityMask(handle,ctypes.byref(process),ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
        if process.value!=mask:raise RuntimeError('Affinity was not applied')
        return available[:count]
    raise RuntimeError('No supported CPU affinity API; do not label thread count as a core limit')


def benchmark(data,images,output,cpus=2,repeats=3):
    affinity=bind_cpus(cpus);started=time.perf_counter();engine=Engine(data,threads=cpus)
    startup=(time.perf_counter()-started)*1000;startup_memory=memory_usage()
    decoded=[(p,cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_COLOR)) for p in images]
    if any(im is None for _,im in decoded):raise ValueError('Unreadable benchmark image')
    for path,image in decoded:engine.scan(image,path.name)
    records=[]
    for repeat in range(repeats):
        for path,image in decoded:
            cpu_start=time.process_time()
            result=engine.scan(image,path.name)
            records.append({'file':path.name,'repeat':repeat,'pixels':[image.shape[1],image.shape[0]],
                'cards':len(result['cards']),'unidentified':len(result['unidentified']),'elapsed_ms':result['elapsed_ms'],
                'process_cpu_ms':round((time.process_time()-cpu_start)*1000,2)})
    times=[r['elapsed_ms'] for r in records]
    report={'logical_cpu_affinity':affinity,'threads':cpus,'platform':platform.platform(),
        'processor':platform.processor(),'host_logical_cpus':os.cpu_count(),
        'model_manifest_sha256':hashlib.sha256((Path(data)/'models/recognition.json').read_bytes()).hexdigest(),
        'startup_ms':round(startup,2),'startup_memory':startup_memory,'final_memory':memory_usage(),
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in images},
        'warmup':'one pass of all images','repeats':repeats,
        'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
        'min_ms':min(times),'max_ms':max(times),'measurements':records,
        'scope':'Same process constrained to listed logical CPUs. Scan includes localization, identity retrieval and field and rank reading; excludes file decoding and HTTP. Other host workloads are not controlled.'}
    write(output,report);print({k:v for k,v in report.items() if k!='measurements'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',type=int,default=2);p.add_argument('--repeats',type=int,default=3);p.add_argument('images',nargs='+',type=Path)
    a=p.parse_args()
    if a.cpus<1 or a.repeats<1:p.error('cpus and repeats must be positive')
    benchmark(a.data,a.images,a.output,a.cpus,a.repeats)
