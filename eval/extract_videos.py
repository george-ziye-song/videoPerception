import zipfile, glob, os, sys, time
def extract(zpath, dest, tag):
    t0=time.time()
    with zipfile.ZipFile(zpath) as zf:
        members=zf.namelist(); n=len(members)
        done=0
        for m in members:
            zf.extract(m, dest)
            done+=1
            if done % 2000 == 0:
                print(f"[{tag}] {os.path.basename(zpath)} {done}/{n}", flush=True)
    print(f"[{tag}] DONE {os.path.basename(zpath)} n={n} {time.time()-t0:.0f}s -> {dest}", flush=True)

# MVBench -> /root/benchmarks/MVBench_video/
mvsrc='/remote-home/ziyesong/videoPerception/data/benchmarks/MVBench/video'
mvdst='/root/benchmarks/MVBench_video'
os.makedirs(mvdst, exist_ok=True)
for z in sorted(glob.glob(mvsrc+'/*.zip')):
    extract(z, mvdst, 'MV')
open(mvdst+'/.extract_done','w').write('ok')

# TemporalBench -> in place (/root/benchmarks/TemporalBench/) so short_video/ long_video/ resolve
tbdst='/root/benchmarks/TemporalBench'
for z in sorted(glob.glob(tbdst+'/*.zip')):
    extract(z, tbdst, 'TB')
open(tbdst+'/.extract_done','w').write('ok')
print("ALL EXTRACT DONE", flush=True)
