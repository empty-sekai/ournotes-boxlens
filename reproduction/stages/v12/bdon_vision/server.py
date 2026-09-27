import base64
import json
import threading
import hashlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .assets import read
from .engine import Engine,merge


def serve(data,port,host='127.0.0.1'):
    data=Path(data)
    engine=Engine(data,threads=2)
    lock=threading.Lock()
    assets={'/art/'+Path(c['file']).name:data/c['file'] for c in engine.cards}
    html=(Path(__file__).parent/'web.html').read_bytes()
    javascript=(Path(__file__).parent/'web.js').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self,body,status=200,content_type='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            if self.path=='/':self.send(html,content_type='text/html; charset=utf-8')
            elif self.path=='/app.js':self.send(javascript,content_type='application/javascript; charset=utf-8')
            elif self.path=='/api/catalog':self.send(engine.cards)
            elif self.path in assets:self.send(assets[self.path].read_bytes(),content_type='image/webp')
            elif self.path=='/health':self.send({'status':'ok','cards':len(engine.cards),'threads':2})
            else:self.send({'error':'Not found'},404)

        def do_POST(self):
            if self.path!='/api/scan':return self.send({'error':'Not found'},404)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<48*1024*1024:raise ValueError('图片总大小超出限制')
                request=json.loads(self.rfile.read(size))
                if not isinstance(request,dict):raise ValueError('请求必须是对象')
                if set(request)-{'schema','player','images'}:raise ValueError('请求含未知字段')
                if request.get('schema','ournotes-boxlens.scan-request/1')!='ournotes-boxlens.scan-request/1':raise ValueError('不支持的请求版本')
                player=request.get('player','local')
                if not isinstance(player,str) or len(player)>100:raise ValueError('玩家标识须为不超过 100 字符的字符串')
                files=request.get('images',[])
                if not isinstance(files,list) or not 1<=len(files)<=30:raise ValueError('每次上传 1–30 张截图')
                scans=[];cache={}
                with lock:
                    for i,file in enumerate(files):
                        if not isinstance(file,dict) or not isinstance(file.get('data'),str):raise ValueError('图片数据格式错误')
                        if set(file)-{'name','data'}:raise ValueError('图片记录含未知字段')
                        source=file.get('name',f'image-{i+1}')
                        if not isinstance(source,str) or not 1<=len(source)<=150:raise ValueError('图片名称须为 1–150 字符的字符串')
                        raw=base64.b64decode(file['data'].split(',')[-1],validate=True)
                        digest=hashlib.sha256(raw).hexdigest()
                        if digest in cache:
                            scan=copy.deepcopy(cache[digest]);scan['source']=source;scan['elapsed_ms']=0.;scan['duplicate_image']=True
                            scans.append(scan);continue
                        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                        if image is None or image.shape[0]*image.shape[1]>20_000_000:
                            raise ValueError('无法读取图片，或分辨率超过 2000 万像素')
                        scan=engine.scan(image,source);cache[digest]=scan;scans.append(scan)
                self.send({'schema':'ournotes-boxlens.scan-result/1','box':merge(scans,player=player),'scans':scans})
            except (ValueError,KeyError,TypeError) as error:
                self.send({'error':str(error)},400)

        def log_message(self,*args):pass

    print(f'Box vision listening on http://{host}:{port}',flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()
