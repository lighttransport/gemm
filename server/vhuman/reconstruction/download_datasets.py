"""Resumable research evaluation starter downloads, outside the repository.

Original stdlib implementation. No dataset scripts are downloaded/executed.
Multiface: official two-expression mini selection, publisher MD5 verification.
SpeakingFaces: one 72-frame RGB/audio utterance read by bounded HTTP ZIP ranges.
Emily: geometry/maps/calibration/polarized references. Gated datasets require
authorized URLs supplied via a local access file; no registration is submitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

DEFAULT_ROOT = Path('/mnt/nvme02/data/vhuman')
MULTIFACE = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/v0.0/identities/6795937/'
EMILY = 'https://vgl.ict.usc.edu/Data/DigitalEmily2/Data/'
HF_REVISION = 'eb2f82264d967f761981a683ecae6992a930dfa2'
SPEAKING = f'https://huggingface.co/datasets/issai/Speaking_Faces/resolve/{HF_REVISION}/image_audio/sub_1_ia.zip'
GATED = ('nersemble','now','facescape','faceolat')
DATASETS = ('multiface','speakingfaces','emily')+GATED
USER_AGENT = 'vhuman-evaluation-downloader/1.0'
RESERVE = 4*1024**3
CHUNK = 1024**2
RANGE_BLOCK = 8*CHUNK
CANCEL = threading.Event()


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(CHUNK),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name+'.writing')
    with partial.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n')
    partial.replace(path)


def acquire_lock(root):
    handle=(Path(root)/'.download.lock').open('a')
    try:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ValueError('another downloader is using this destination') from None
    return handle


def open_url(url, headers=None):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1')):
        raise ValueError('dataset URL must use HTTPS')
    request = urllib.request.Request(url,headers={'User-Agent':USER_AGENT,'Accept-Encoding':'identity',**(headers or {})})
    return urllib.request.urlopen(request,timeout=60)


def metadata(url):
    with open_url(url) as response:
        if int(response.headers.get('Content-Length',0))>4*CHUNK:
            raise ValueError('metadata response exceeds limit')
        data = response.read(4*CHUNK+1)
    if len(data)>4*CHUNK:
        raise ValueError('metadata response exceeds limit')
    return data


def download(url, path, *, checksum=None, algorithm='sha256', max_bytes=32*1024**3):
    """Resume only with a matching validator and an exact Content-Range."""
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    receipt = path.with_name(path.name+'.receipt.json')
    if path.exists():
        actual = digest(path,algorithm)
        if checksum and actual!=checksum:
            raise ValueError('existing archive checksum mismatch: '+path.name)
        if checksum or (receipt.exists() and json.loads(receipt.read_text()).get('sha256')==digest(path)):
            return dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path),status='verified existing',publisher_checksum=checksum)
        raise ValueError('existing file lacks a valid receipt: '+path.name)
    partial = path.with_name(path.name+'.part')
    if partial.with_name(partial.name+'.aria2').exists():
        raise ValueError('aria2-managed partial needs --engine aria2 to resume: '+path.name)
    state = path.with_name(path.name+'.part.json')
    source_hash = hashlib.sha256(url.encode()).hexdigest()  # never store signed URL
    last_progress = 0.
    attempt=0
    while attempt<4:
        try:
            previous = json.loads(state.read_text()) if state.exists() else {}
            offset = partial.stat().st_size if partial.exists() else 0
            if offset and (previous.get('source_hash')!=source_hash or not previous.get('validator')):
                partial.unlink();offset=0
            requested_end=offset+RANGE_BLOCK-1
            headers = {'Range':f'bytes={offset}-{requested_end}'}
            if offset:headers['If-Range']=previous['validator']
            with open_url(url,headers) as response:
                if 'text/html' in response.headers.get('Content-Type','').lower():
                    raise ValueError('download endpoint returned HTML, not dataset bytes')
                validator = response.headers.get('ETag') or response.headers.get('Last-Modified')
                if offset and response.status==200:
                    # Range ignored or source changed: restart, never concatenate.
                    offset=0
                content_range = response.headers.get('Content-Range','')
                if response.status==206:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content_range)
                    if not match or int(match[1])!=offset:
                        raise ValueError('invalid resume Content-Range')
                    total = int(match[3])
                    response_end=int(match[2])+1
                    if response_end>total or response_end>requested_end+1 or response_end<=offset:
                        raise ValueError('invalid bounded response range')
                    if offset and validator!=previous.get('validator'):
                        raise ValueError('resumed dataset validator changed')
                else:
                    length = response.headers.get('Content-Length')
                    total = int(length) if length else None
                    response_end=total
                if total is not None and (total>max_bytes or total<offset):
                    raise ValueError('archive size outside configured limit')
                remaining = (total-offset) if total is not None else 0
                if shutil.disk_usage(path.parent).free<remaining+RESERVE:
                    raise ValueError('insufficient disk space with 4 GiB reserve')
                atomic_json(state,dict(source_hash=source_hash,validator=validator,total=total))
                mode='ab' if offset else 'wb'
                with partial.open(mode) as f:
                    while block:=response.read(CHUNK):
                        if CANCEL.is_set():raise RuntimeError('download interrupted; partial retained')
                        if offset+len(block)>max_bytes or (total is not None and offset+len(block)>total):
                            raise ValueError('archive exceeded declared or configured size')
                        if shutil.disk_usage(path.parent).free<len(block)+RESERVE:
                            raise ValueError('disk reserve reached')
                        f.write(block);offset+=len(block)
                        if time.monotonic()-last_progress>30:
                            print(f'{path.name}: {offset/1024**3:.2f} GiB'+(f' / {total/1024**3:.2f}' if total else ''),flush=True)
                            last_progress=time.monotonic()
                    f.flush();os.fsync(f.fileno())
                if response_end is not None and offset!=response_end:
                    raise OSError('incomplete archive response')
            if total is not None and offset<total:
                attempt=0
                continue
            actual = digest(partial,algorithm)
            if checksum and actual.lower()!=checksum.lower():
                partial.unlink();state.unlink(missing_ok=True)
                raise ValueError('publisher checksum mismatch: '+path.name)
            result=dict(file=path.name,bytes=offset,sha256=digest(partial),publisher_checksum=checksum,status='downloaded')
            partial.replace(path);atomic_json(receipt,result);state.unlink(missing_ok=True)
            return result
        except urllib.error.HTTPError as exc:
            if exc.code not in (408,429,500,502,503,504) or attempt==3:
                raise RuntimeError(f'HTTP {exc.code} downloading {path.name}') from None
            attempt+=1;time.sleep(min(10,2**attempt))
        except (OSError,urllib.error.URLError) as exc:
            if attempt==3:
                raise RuntimeError('network transfer failed: '+path.name) from None
            attempt+=1;time.sleep(min(10,2**attempt))
    raise RuntimeError('download retry limit reached')


def download_multiface_aria2(url,path,checksum):
    """Optional installed aria2 handles parallel ranges; publisher MD5 is mandatory.

    Only the public Multiface endpoint uses this backend, never private URLs.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return download(url,path,checksum=checksum,algorithm='md5')
    with open_url(url,{'Range':'bytes=0-0'}) as response:
        match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
        if response.status!=206 or not match:raise ValueError('Multiface size probe failed')
        total=int(match[1]);response.read(1)
    if total>32*1024**3 or shutil.disk_usage(path.parent).free<total+RESERVE:
        raise ValueError('Multiface size/free-space limit reached')
    partial=path.with_name(path.name+'.part')
    command=['aria2c','--continue=true','--auto-file-renaming=false','--allow-overwrite=true',
             '--file-allocation=none','--check-integrity=true','--checksum=md5='+checksum,
             '--max-connection-per-server=8','--split=8','--min-split-size=8M',
             '--max-tries=4','--retry-wait=3','--timeout=60','--connect-timeout=20',
             '--summary-interval=30','--console-log-level=warn','--download-result=hide',
             '--dir='+str(path.parent),'--out='+partial.name,url]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        for line in process.stdout:
            if CANCEL.is_set():
                process.terminate();raise RuntimeError('download interrupted; aria2 partial retained')
            if '://' not in line and (line.startswith('[#') or 'ERROR' in line):
                print(path.name+': '+line.strip(),flush=True)
        if process.wait()!=0:raise RuntimeError('aria2 transfer failed; partial retained: '+path.name)
    finally:
        if process.poll() is None:process.terminate();process.wait()
        process.stdout.close()
    if not partial.exists() or partial.stat().st_size!=total or digest(partial,'md5')!=checksum:
        raise ValueError('Multiface aria2 archive failed publisher verification')
    result=dict(file=path.name,bytes=total,sha256=digest(partial),publisher_checksum=checksum,
                status='downloaded',engine='aria2 parallel ranges')
    partial.replace(path)
    atomic_json(path.with_name(path.name+'.receipt.json'),result)
    path.with_name(path.name+'.part.json').unlink(missing_ok=True)
    return result


def extract_archive(path, destination):
    """Extract regular files only; reject traversal/links, stage before publish."""
    path,destination = Path(path),Path(destination)
    stamp = destination.with_name(destination.name+'.extracted.json')
    checksum = digest(path)
    if destination.exists():
        if stamp.exists() and json.loads(stamp.read_text()).get('sha256')==checksum:
            return dict(folder=destination.name,status='existing extraction')
        raise ValueError('existing extraction has no matching completion record')
    staging = destination.with_name('.'+destination.name+'.partial')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    def target(name):
        rel=Path(name)
        if '\\' in name or rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('unsafe archive path')
        out=staging/rel
        if not out.resolve().is_relative_to(staging.resolve()):
            raise ValueError('unsafe archive destination')
        return out
    def copy(src,out,size):
        if shutil.disk_usage(staging).free<size+RESERVE:
            raise ValueError('insufficient extraction disk space')
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('wb') as f:
            copied=0
            while block:=src.read(CHUNK):
                copied+=len(block)
                if copied>size:
                    raise ValueError('archive member exceeds declared size')
                f.write(block)
            if copied!=size:
                raise ValueError('truncated archive member')
    files=0
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                for member in z.infolist():
                    out=target(member.filename)
                    if member.is_dir():
                        out.mkdir(parents=True,exist_ok=True);continue
                    if (member.external_attr>>16)&0o170000==0o120000:
                        raise ValueError('archive symlink forbidden')
                    with z.open(member) as src:copy(src,out,member.file_size)
                    files+=1
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as t:
                for member in t:
                    if member.isdir() and member.name in ('.','./'):
                        continue
                    out=target(member.name)
                    if member.isdir():
                        out.mkdir(parents=True,exist_ok=True);continue
                    if not member.isfile():
                        raise ValueError('archive links/special files forbidden')
                    with t.extractfile(member) as src:copy(src,out,member.size)
                    files+=1
        elif path.suffix.lower()=='.rar':
            try:
                import rarfile
            except ImportError:
                return dict(status='archive retained; safe RAR extraction needs rarfile==4.2 and unrar/7z')
            with rarfile.RarFile(path) as archive:
                for member in archive.infolist():
                    out=target(member.filename)
                    if member.is_symlink() or getattr(member,'file_redir',None):
                        raise ValueError('RAR links forbidden')
                    if member.isdir():
                        out.mkdir(parents=True,exist_ok=True);continue
                    if not member.is_file():
                        raise ValueError('RAR special files forbidden')
                    with archive.open(member) as src:copy(src,out,member.file_size)
                    files+=1
        else:
            return dict(status='unrecognized archive retained; not extracted')
        staging.replace(destination)
        atomic_json(stamp,dict(sha256=checksum,files=files))
        return dict(folder=destination.name,files=files,status='extracted')
    finally:
        if staging.exists():shutil.rmtree(staging)


def multiface(folder, extract=True, workers=2, engine='auto'):
    folder.mkdir(parents=True,exist_ok=True)
    raw = metadata(MULTIFACE+'CHECKSUM')
    (folder/'CHECKSUM').write_bytes(raw)
    checks={line.split()[-1].lstrip('*'):line.split()[0] for line in raw.decode().splitlines() if len(line.split())==2}
    names=['metadata.tar','audio.tar']+[f'{kind}--{expression}.tar' for expression in ('E057_Cheeks_Puffed','E061_Lips_Puffed')
            for kind in ('tracked_mesh','unwrapped_uv_1024','images')]
    rows=[]
    def transfer(name):
        if name not in checks or not re.fullmatch('[0-9a-fA-F]{32}',checks[name]):
            raise ValueError('publisher MD5 missing for '+name)
        use_aria2=engine=='aria2' or (engine=='auto' and shutil.which('aria2c'))
        if use_aria2 and not shutil.which('aria2c'):raise ValueError('aria2c is not installed')
        result=download_multiface_aria2(MULTIFACE+name,folder/'archives'/name,checks[name]) if use_aria2 else download(
            MULTIFACE+name,folder/'archives'/name,checksum=checks[name],algorithm='md5')
        if extract:result['extraction']=extract_archive(folder/'archives'/name,folder/'extracted'/Path(name).stem)
        return result
    pool=ThreadPoolExecutor(max_workers=workers)
    failures=[]
    try:
        futures={pool.submit(transfer,name):name for name in names}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
                atomic_json(folder/'progress.json',sorted(rows,key=lambda row:row['file']))
            except Exception:
                failures.append(futures[future])
    except KeyboardInterrupt:
        CANCEL.set();raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    if failures:raise RuntimeError('Multiface transfers failed; rerun to resume: '+', '.join(failures))
    rows.sort(key=lambda row:row['file'])
    return dict(status='complete',license='CC-BY-NC-4.0',subject='6795937',expressions=['E057_Cheeks_Puffed','E061_Lips_Puffed'],assets=rows)


def emily(folder, extract=True):
    names=['2.1/Emily_2_1_OBJ.zip','2.1/Emily_2_1_Textures.zip','DigitalEmily2_Calibration.rar',
           'DigitalEmily2_Unpolarized.rar','DigitalEmily2_FlashParallel.rar','DigitalEmily2_FlashCross.rar','DigitalEmily2_SpecularOnly.rar']
    rows=[]
    for name in names:
        result=download(EMILY+name,folder/'archives'/Path(name).name)
        rows.append(result);atomic_json(folder/'progress.json',rows)
        if extract:result['extraction']=extract_archive(folder/'archives'/Path(name).name,folder/'extracted'/Path(name).stem)
    return dict(status='complete',license='research/illustration, noncommercial, no redistribution',assets=rows)


class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP range reader; never buffers the full subject archive."""
    def __init__(self,url):
        self.url=url;self.position=0;self.cache=b'';self.start=0
        with open_url(url,{'Range':'bytes=0-0'}) as r:
            match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''))
            if r.status!=206 or not match:
                raise ValueError('SpeakingFaces endpoint does not support bounded ZIP ranges')
            self.size=int(match[1]);r.read(1)
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        value=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if value<0:raise ValueError('negative archive seek')
        self.position=value;return value
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        size=min(size,self.size-self.position)
        if size<=0:return b''
        if size>8*CHUNK:raise ValueError('ZIP range request exceeds 8 MiB bound')
        if not (self.start<=self.position and self.position+size<=self.start+len(self.cache)):
            end=min(self.size-1,self.position+max(size,CHUNK)-1)
            with open_url(self.url,{'Range':f'bytes={self.position}-{end}'}) as r:
                expected=f'bytes {self.position}-{end}/{self.size}'
                if r.status!=206 or r.headers.get('Content-Range')!=expected:
                    raise ValueError('invalid ZIP range response')
                self.cache=r.read(end-self.position+2)
                if len(self.cache)!=end-self.position+1:raise ValueError('incomplete ZIP range')
                self.start=self.position
        begin=self.position-self.start;self.position+=size
        return self.cache[begin:begin+size]


def speakingfaces(folder):
    folder.mkdir(parents=True,exist_ok=True)
    prefix='sub_1_ia/trial_1/rgb_image_cmd/1_1_2_7_611_'
    rows=[]
    with RemoteZipFile(SPEAKING) as remote,zipfile.ZipFile(remote) as z:
        frames=sorted([i for i in z.infolist() if i.filename.startswith(prefix) and i.filename.endswith('_2.png')],
                      key=lambda i:int(i.filename.removesuffix('_2.png').split('_')[-1]))
        if len(frames)!=72:raise ValueError('unexpected SpeakingFaces sample frame count')
        audio=z.getinfo('sub_1_ia/trial_1/mic1_audio_cmd_trim/1_1_2_7_611_1.wav')
        for index,item in enumerate(frames+[audio]):
            if item.file_size>16*CHUNK:raise ValueError('SpeakingFaces sample member too large')
            dst=folder/('audio.wav' if item is audio else f'frames/{index:05d}.png')
            dst.parent.mkdir(parents=True,exist_ok=True)
            receipt=dst.with_name(dst.name+'.receipt.json')
            if dst.exists() and receipt.exists() and digest(dst)==json.loads(receipt.read_text()).get('sha256'):
                row=json.loads(receipt.read_text())
            else:
                part=dst.with_name(dst.name+'.part')
                with z.open(item) as src,part.open('wb') as out:
                    shutil.copyfileobj(src,out,CHUNK)
                if part.stat().st_size!=item.file_size:raise ValueError('ZIP member size mismatch')
                part.replace(dst)
                row=dict(file=str(dst.relative_to(folder)),bytes=item.file_size,sha256=digest(dst),zip_crc32=f'{item.CRC:08x}')
                atomic_json(receipt,row)
            rows.append(row)
            if index%12==0:
                print(f'SpeakingFaces: {min(index+1,72)}/72 frames',flush=True)
    return dict(status='complete',license='CC-BY-4.0',source_url=SPEAKING,revision=HF_REVISION,
                sample='1_1_2_7_611',fps=28,frames=72,assets=rows)


def gated(name,folder,access,extract):
    assets=access.get(name,[])
    if not assets:
        return dict(status='needs approved access',reason='provide authorized archive URLs through --access-file; see evaluation_sources.json')
    if not isinstance(assets,list):raise ValueError('access-file entries must be lists')
    rows=[]
    for asset in assets:
        filename=asset['name']
        if not isinstance(filename,str) or Path(filename).name!=filename or filename in ('.','..') or '\\' in filename:
            raise ValueError('invalid private asset filename')
        checksum=asset.get('sha256') or asset.get('md5')
        algorithm='sha256' if asset.get('sha256') else 'md5' if asset.get('md5') else 'sha256'
        result=download(asset['url'],folder/'archives'/filename,checksum=checksum,algorithm=algorithm)
        rows.append(result);atomic_json(folder/'progress.json',rows)
        if extract:result['extraction']=extract_archive(folder/'archives'/filename,folder/'extracted'/Path(filename).stem)
    return dict(status='complete',assets=rows,license='approved dataset-specific access terms; no access URLs recorded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS[:3]),help='default: public starters only')
    parser.add_argument('--access-file',type=Path,help='private JSON: dataset -> [{name,url,optional sha256/md5}]')
    parser.add_argument('--no-extract',action='store_true')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=2,help='concurrent Multiface archive transfers (1..4)')
    parser.add_argument('--engine',choices=['auto','urllib','aria2'],default='auto',help='Multiface backend; auto prefers installed aria2c')
    parser.add_argument('--accept-research-terms',action='store_true',help='acknowledge listed research-only/media terms')
    args=parser.parse_args()
    if not args.accept_research_terms:parser.error('review catalog terms and pass --accept-research-terms')
    repo=Path(__file__).resolve().parents[3]
    if args.root.resolve().is_relative_to(repo):parser.error('datasets must be stored outside the repository')
    args.root.mkdir(parents=True,exist_ok=True)
    lock=acquire_lock(args.root)  # held until main exits; OS releases on interruption
    access=json.loads(args.access_file.read_text()) if args.access_file else {}
    catalog=json.loads((Path(__file__).parent/'data/evaluation_sources.json').read_text())
    old=args.root/'download_manifest.json'
    previous=json.loads(old.read_text()) if old.exists() else {}
    report=dict(format='vhuman.dataset_download.v1',selection=args.datasets,root=str(args.root.resolve()),
                datasets=previous.get('datasets',{}),catalog=catalog)
    failures=0
    for name in args.datasets:
        print('Starting '+name,flush=True)
        folder=args.root/name;folder.mkdir(parents=True,exist_ok=True)
        try:
            result=multiface(folder,not args.no_extract,args.workers,args.engine) if name=='multiface' else emily(folder,not args.no_extract) if name=='emily' else speakingfaces(folder) if name=='speakingfaces' else gated(name,folder,access,not args.no_extract)
        except Exception as exc:
            # Exception strings can contain signed URLs; only our controlled
            # validation errors are retained. Never log URLs from access files.
            reason=str(exc) if isinstance(exc,(ValueError,RuntimeError)) and '://' not in str(exc) else type(exc).__name__
            result=dict(status='failed; rerun to resume',reason=reason);failures+=1
        report['datasets'][name]=result
        atomic_json(folder/'download_manifest.json',result)
        atomic_json(args.root/'download_manifest.json',report)
        print(name+': '+result['status'],flush=True)
    pending=any(report['datasets'][name]['status']=='needs approved access' for name in args.datasets)
    lock.close()
    raise SystemExit(1 if failures else 2 if pending else 0)


if __name__=='__main__':main()
