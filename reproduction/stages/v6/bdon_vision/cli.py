import argparse
import json
from pathlib import Path

import cv2

from .assets import prepare,write


def main():
    parser=argparse.ArgumentParser(description='CPU-only BDON screenshot importer')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data',type=Path,required=True)
    p=sub.add_parser('scan');p.add_argument('--data',type=Path,required=True)
    p.add_argument('images',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    p=sub.add_parser('serve');p.add_argument('--data',type=Path,required=True);p.add_argument('--port',type=int,default=18776)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.data)
    elif args.command=='scan':
        from .engine import Engine,merge
        engine=Engine(args.data,args.threads)
        scans=[]
        for path in args.images:
            im=cv2.imread(str(path))
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
            cv2.imwrite(str(args.output/(path.stem+'-annotated.jpg')),preview)
            print(json.dumps({'source':path.name,'cards':len(result['cards']),'elapsed_ms':result['elapsed_ms']},ensure_ascii=False),flush=True)
        write(args.output/'observations.json',scans)
        write(args.output/'box.json',merge(scans))
    elif args.command=='serve':
        from .server import serve
        serve(args.data,args.port)
