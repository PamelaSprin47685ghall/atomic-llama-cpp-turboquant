#!/usr/bin/env python3
"""Read-only audit of actual tensor families and independent micro measurements."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--metadata',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
records=json.loads(a.metadata.read_text())
families=Counter()
for record in records[:6]:
    for t in record['tensors']:
        families[(re.sub(r'^blk\.\d+\.','',t['name']),tuple(t['shape']),t['raw_type'],t['storage_extent_bytes'])]+=1
for (name,shape,dtype,extent),count in sorted(families.items()):
    print(json.dumps({'family':name,'count':count,'shape':shape,'type':dtype,'storage_bytes':extent}))
print('INDEPENDENT_MEASUREMENTS')
for pattern in ('norm-geometry-*.log','command-floor-*.log','q-validated-*.log','lo-*.log'):
    for f in sorted(a.root.glob(pattern)):
        data=[json.loads(l) for l in f.read_text().splitlines() if l.startswith('{')]
        if data:print(json.dumps({'source':str(f),'result':data[-1]}))
