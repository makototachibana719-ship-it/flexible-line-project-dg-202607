#!/usr/bin/env python3
"""Build an immutable virtual index over existing local/corrective pairs."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path('/home/makoto/flexible_line_project')
SOURCES={'round1_corrective':ROOT/'results/datasets/peg_v2_corrective_postbc_round1_20260908_001/corrective_round1.pkl','local_insertion':ROOT/'results/datasets/peg_v2_local_insertion_supervision_20260908_001/local_insertion_supervision.pkl'}
OUT=ROOT/'results/datasets/peg_v2_local_retention_anchor_v1_20260908_001'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise RuntimeError('fresh anchor directory required')
 OUT.mkdir(parents=True)
 unique={}
 for source,path in SOURCES.items():
  rows=pickle.loads(path.read_bytes())
  for i,row in enumerate(rows):
   o=np.asarray(row['observations'],np.float32);a=np.asarray(row['actions'],np.float32)
   key=hashlib.sha256(o.tobytes()+a.tobytes()).hexdigest()
   if key not in unique:unique[key]={'pair_sha256':key,'primary_source':source,'primary_row':i,'duplicates':[]}
   else:unique[key]['duplicates'].append({'source':source,'row':i})
 entries=sorted(unique.values(),key=lambda x:x['pair_sha256'])
 index=OUT/'virtual_index.json';index.write_text(json.dumps(entries,indent=2,sort_keys=True)+'\n')
 manifest={'name':'LOCAL_RETENTION_ANCHOR_V1','storage':'VIRTUAL_REFERENCES_ONLY','production_demo_included':False,'sources':{k:{'path':str(v),'sha256':sha(v),'rows':len(pickle.loads(v.read_bytes()))} for k,v in SOURCES.items()},'input_pairs':sum(len(pickle.loads(v.read_bytes())) for v in SOURCES.values()),'unique_pairs':len(entries),'deduplicated_pairs':sum(1+len(x['duplicates']) for x in entries)-len(entries),'virtual_index':str(index),'LOCAL_RETENTION_ANCHOR_SHA256':sha(index)}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');(OUT/'sha256.txt').write_text(manifest['LOCAL_RETENTION_ANCHOR_SHA256']+'  virtual_index.json\n');print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
