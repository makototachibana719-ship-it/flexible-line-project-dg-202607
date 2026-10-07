#!/usr/bin/env python3
import sys,pickle,numpy as np
from pathlib import Path
p=Path(sys.argv[1])
with p.open("rb") as f: obj=pickle.load(f)
def walk(x,key="root",indent=0,depth=0):
    pre="  "*indent
    if depth>6: print(pre+key+": <max depth>"); return
    if isinstance(x,dict):
        print(f"{pre}{key}: dict keys={list(x.keys())}")
        for k,v in x.items(): walk(v,str(k),indent+1,depth+1)
    elif isinstance(x,(list,tuple)):
        print(f"{pre}{key}: {type(x).__name__} len={len(x)}")
        if x: walk(x[0],"[0]",indent+1,depth+1)
    else:
        try:
            a=np.asarray(x); print(f"{pre}{key}: type={type(x).__name__} shape={a.shape} dtype={a.dtype}")
        except Exception: print(f"{pre}{key}: type={type(x).__name__}")
print(f"file={p}"); print(f"top_type={type(obj).__name__}")
try: print(f"top_len={len(obj)}")
except Exception: pass
walk(obj)
