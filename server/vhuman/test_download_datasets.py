"""HTTP resume/integrity and archive containment checks for dataset downloads."""
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .reconstruction import download_datasets as d

ROOT=Path(os.environ.get('VHUMAN_DATASET_TEST_ROOT',
    str(Path(__file__).resolve().parents[2]/'tmp/test-dataset-download'))).absolute()
ROOT.mkdir(parents=True,exist_ok=True)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT)
        self.root=Path(self.temp.name)
        self.body=b'original fixture archive bytes'*1000
        self.requests=[]
        test=self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                test.requests.append(self.headers.get('Range'))
                offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
                ranged='Range' in self.headers and self.path!='/ignore'
                start=offset if ranged else 0
                requested=self.headers.get('Range','bytes=0-').split('-')[-1]
                end=min(len(test.body),int(requested)+1) if ranged and requested else len(test.body)
                self.send_response(206 if ranged else 200)
                self.send_header('Content-Type','application/octet-stream')
                self.send_header('ETag','"fixture"')
                self.send_header('Content-Length',str(end-start))
                if ranged:
                    self.send_header('Content-Range',f'bytes {start+1 if self.path=="/bad" else start}-{end-1}/{len(test.body)}')
                self.end_headers();self.wfile.write(test.body[start:end])
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()

    def partial(self,url,path):
        path.with_name(path.name+'.part').write_bytes(self.body[:100])
        d.atomic_json(path.with_name(path.name+'.part.json'),dict(source_hash=hashlib.sha256(url.encode()).hexdigest(),validator='"fixture"'))

    def test_resume_and_existing_integrity(self):
        url=self.url+'/archive';path=self.root/'archive.bin';self.partial(url,path)
        row=d.download(url,path,checksum=hashlib.md5(self.body).hexdigest(),algorithm='md5')
        self.assertEqual(len(self.requests),1)
        self.assertTrue(self.requests[0].startswith('bytes=100-'))
        self.assertEqual(path.read_bytes(),self.body)
        self.assertEqual(row['sha256'],hashlib.sha256(self.body).hexdigest())
        self.assertEqual(d.download(url,path)['status'],'verified existing')
        path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):d.download(url,path)

    def test_range_ignored_restarts_without_append(self):
        url=self.url+'/ignore';path=self.root/'archive.bin';self.partial(url,path)
        d.download(url,path)
        self.assertEqual(path.read_bytes(),self.body)

    @unittest.skipUnless(shutil.which('aria2c'), 'optional aria2c unavailable')
    def test_aria2_resume_and_publisher_checksum(self):
        path=self.root/'parallel.bin'
        self.partial(self.url+'/archive',path)
        row=d.download_multiface_aria2(self.url+'/archive',path,hashlib.md5(self.body).hexdigest())
        self.assertEqual(path.read_bytes(),self.body)
        self.assertEqual(row['sha256'],hashlib.sha256(self.body).hexdigest())
        self.assertFalse(path.with_name(path.name+'.part.json').exists())

    def test_bad_range_and_checksum_do_not_publish(self):
        path=self.root/'bad.bin';url=self.url+'/bad';self.partial(url,path)
        with self.assertRaisesRegex(ValueError,'Content-Range'):d.download(url,path)
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError,'checksum'):d.download(self.url+'/whole',self.root/'wrong.bin',checksum='0'*64)
        self.assertFalse((self.root/'wrong.bin').exists())

    def test_zip_containment_and_tar_link(self):
        archive=self.root/'unsafe.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside','bad')
        with self.assertRaisesRegex(ValueError,'unsafe'):d.extract_archive(archive,self.root/'out')
        self.assertFalse((self.root/'outside').exists());self.assertFalse((self.root/'out').exists())
        archive=self.root/'unsafe.tar'
        with tarfile.open(archive,'w') as t:
            item=tarfile.TarInfo('link');item.type=tarfile.SYMTYPE;item.linkname='../outside';t.addfile(item)
        with self.assertRaisesRegex(ValueError,'links'):d.extract_archive(archive,self.root/'out')

    def test_extract_complete_and_gated_status(self):
        archive=self.root/'safe.tar'
        with tarfile.open(archive,'w') as t:
            item=tarfile.TarInfo('identity/KRT');item.size=3;t.addfile(item,io.BytesIO(b'cam'))
        out=self.root/'data'
        self.assertEqual(d.extract_archive(archive,out)['files'],1)
        self.assertEqual((out/'identity/KRT').read_bytes(),b'cam')
        self.assertEqual(d.extract_archive(archive,out)['status'],'existing extraction')
        self.assertEqual(d.gated('now',self.root,{},True)['status'],'needs approved access')
        with self.assertRaises(ValueError):d.gated('now',self.root,{'now':[dict(name='../escape',url=self.url)]},True)

    def test_destination_lock(self):
        with d.acquire_lock(self.root):
            with self.assertRaisesRegex(ValueError,'another downloader'):d.acquire_lock(self.root)
        with d.acquire_lock(self.root):pass

    def test_bounded_segments_publish_only_complete_archive(self):
        from unittest.mock import patch
        path=self.root/'segments.bin'
        with patch.object(d,'RANGE_BLOCK',4096):
            d.download(self.url+'/segments',path,checksum=hashlib.sha256(self.body).hexdigest())
        self.assertGreater(len(self.requests),1)
        self.assertEqual(path.read_bytes(),self.body)


if __name__=='__main__':unittest.main()
