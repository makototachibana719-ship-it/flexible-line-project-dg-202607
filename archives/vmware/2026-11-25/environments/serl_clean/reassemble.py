"""Reassemble and verify the serl_clean archive; never overwrite or extract."""
import csv
import hashlib
import json
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
info=json.loads((base/'ARCHIVE_INFO.json').read_text(encoding='utf-8'))
output=Path(sys.argv[1]) if len(sys.argv)==2 else Path.cwd()/'serl_clean.tar.gz'
if len(sys.argv)>2:
    raise SystemExit('Usage: python3 reassemble.py [new-output-file]')
with (base/'PARTS_SHA256.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==info['parts']
overall=hashlib.sha256()
total=0
with output.open('xb') as sink:
    for index,row in enumerate(rows,1):
        name=row['filename']
        assert name==f'serl_clean.tar.gz.part{index:04d}'
        part_hash=hashlib.sha256()
        count=0
        with (base/name).open('rb') as source:
            for chunk in iter(lambda:source.read(4194304),b''):
                sink.write(chunk)
                part_hash.update(chunk)
                overall.update(chunk)
                count+=len(chunk)
        assert count==int(row['bytes']), 'Part size mismatch: '+name
        assert part_hash.hexdigest()==row['sha256'], 'Part hash mismatch: '+name
        total+=count
assert total==info['archive_bytes'], 'Archive size mismatch'
assert overall.hexdigest()==info['archive_sha256'], 'Archive hash mismatch'
print('Verified archive:',output)
print('SHA256:',overall.hexdigest())
