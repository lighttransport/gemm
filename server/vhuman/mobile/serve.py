"""Serve an exported browser preview over HTTPS using a supplied certificate."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--cert',type=Path,required=True)
    parser.add_argument('--key',type=Path,required=True)
    parser.add_argument('--bind',default='0.0.0.0')
    parser.add_argument('--port',type=int,default=8443)
    args=parser.parse_args()
    root=args.directory.resolve(strict=True)
    if not (root/'index.html').is_file():parser.error('directory must contain the exported index.html')
    for path in (args.cert,args.key):
        if path.resolve().is_relative_to(root):parser.error('keep TLS certificates and private keys outside the served directory')
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(args.cert,args.key)
    with ThreadingHTTPServer((args.bind,args.port),partial(SimpleHTTPRequestHandler,directory=str(root))) as server:
        server.socket=context.wrap_socket(server.socket,server_side=True)
        print(f'Serving {root} at https://{args.bind}:{args.port}/',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=='__main__':main()
