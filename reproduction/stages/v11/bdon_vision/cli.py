import argparse
import json
import sys
from pathlib import Path

import cv2

from .assets import prepare,write


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='CPU-only BDON screenshot importer')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data',type=Path,required=True)
    p.add_argument('--offline',action='store_true',help='Use supplied art only; do not fetch missing images')
    p=sub.add_parser('scan');p.add_argument('--data',type=Path,required=True)
    p.add_argument('images',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    p=sub.add_parser('serve');p.add_argument('--data',type=Path,required=True);p.add_argument('--port',type=int,default=18776)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.data,allow_download=not args.offline)
    elif args.command=='scan':
        from .engine import Engine,merge
        engine=Engine(args.data,args.threads)
        scans=[]
        for index,path in enumerate(args.images):
            import numpy as np
            im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(f'Cannot decode {path}')
            result=engine.scan(im,path.name)
            scans.append(result)
            preview=im.copy()
            for card in result['cards']:
                x,y,w,h=map(round,card['bbox'])
                cv2.rectangle(preview,(x,y),(x+w,y+h),(60,210,30),2)
                label=f"{card['kind']}#{card['id']} Lv.{card['level']['value']} R{card['card_rank']['value']}"
                cv2.putText(preview,label,(x,max(14,y+16)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3)
                cv2.putText(preview,label,(x,max(14,y+16)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            args.output.mkdir(parents=True,exist_ok=True)
            preview_name=f"{index+1:03d}-{result['source_id'][:12]}-annotated.jpg"
            cv2.imencode('.jpg',preview)[1].tofile(str(args.output/preview_name))
            result['annotated_file']=preview_name
            print(json.dumps({'source':path.name,'cards':len(result['cards']),'elapsed_ms':result['elapsed_ms']},ensure_ascii=False),flush=True)
        write(args.output/'observations.json',scans)
        write(args.output/'box.json',merge(scans))
    elif args.command=='serve':
        from .server import serve
        serve(args.data,args.port)
