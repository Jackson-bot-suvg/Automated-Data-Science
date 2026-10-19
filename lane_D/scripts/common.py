from pathlib import Path
import csv, hashlib, json, os
PKG = Path(__file__).resolve().parents[1]
BASE_SHA = 'b9dbebed269f62767e149b93a02f71051efc1027'
def base():
    configured=os.environ.get('LANE_D_BASE')
    candidates=[Path(configured)] if configured else [PKG.parent,PKG.parent/'Automated-Data-Science-main']
    for candidate in candidates:
        p=candidate.resolve()
        if (p/'app').is_dir() and (p/'data').is_dir():return p
    raise RuntimeError('Set LANE_D_BASE to the supplied Automated-Data-Science-main directory')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def readcsv(p):
    if not Path(p).exists(): return []
    with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ['status'])
    tmp=p.with_suffix(p.suffix+'.tmp')
    with open(tmp,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
    tmp.replace(p)
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
