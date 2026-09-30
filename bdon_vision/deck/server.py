"""Local integration UI; existing BoxLens HTTP server handles all recognition."""
import base64, json, mimetypes, subprocess, sys, threading, time, uuid, socket
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.parse import unquote
from .common import read, write, digest, strict_json
from .adapter import inventory,adapt,IncompleteInventory

def serve(run,port=18790,solver_bin=None,boxport=18793,*,data,deck_data):
    data=Path(data).resolve();deck_data=Path(deck_data).resolve()
    from .visual_data import binding
    loaded_visual=binding(data,deck_data)
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True);sessions={'showcase':run};lock=threading.Lock()
    for check_port in (port,boxport):
        with socket.socket() as probe:
            try:probe.bind(('127.0.0.1',check_port))
            except OSError as e:raise RuntimeError(f'Port {check_port} already occupied; choose isolated UI and BoxLens ports') from e
    logs=run/'server-logs';logs.mkdir(exist_ok=True)
    log=(logs/('boxlens-'+uuid.uuid4().hex+'.log')).open('w',encoding='utf-8')
    worker=subprocess.Popen([sys.executable,'-m','bdon_vision','serve','--data',str(data),'--port',str(boxport)],
        stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        for _ in range(100):
            if worker.poll() is not None: raise RuntimeError(f'BoxLens HTTP worker exited; inspect {log.name}')
            try:
                with urlopen(f'http://127.0.0.1:{boxport}/health',timeout=.2) as r:
                    if json.load(r)['status']=='ok':break
            except OSError: time.sleep(.1)
        else:raise RuntimeError('BoxLens worker failed to become ready')
        class Handler(BaseHTTPRequestHandler):
            def send(self,obj,status=200,mime='application/json; charset=utf-8'):
                body=obj if isinstance(obj,bytes) else json.dumps(obj,ensure_ascii=False).encode('utf-8')
                self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
            def do_GET(self):
                try:
                    if self.path=='/':return self.send(Path(__file__).with_name('web.html').read_bytes(),mime='text/html; charset=utf-8')
                    if self.path=='/health':return self.send({'status':'ok','ports':[port,boxport],'run':str(run),'solver':str(solver_bin) if solver_bin else None,'deckDataSha256':loaded_visual['boundDeckDataSha256'],'visualManifestSha256':loaded_visual['datasetManifestSha256'],'visualIdentityCounts':loaded_visual['identityCounts']})
                    if self.path=='/api/run':
                        d={"boxLensUrl":f"http://127.0.0.1:{boxport}/","loadedVisualData":loaded_visual}
                        for key,name in [('inventory','inventory.json'),('adaptation','adaptation.json'),('roster','roster.json'),('manifest','fixtures/fixture-manifest.json'),('scan','scan/box.json'),('baseline','baseline-live.stdout.json'),('metrics','e2e-report.json'),('matrix','scene-matrix.json'),('recommendations','recommendations-final/recommendations.json' if (run/'recommendations-final/recommendations.json').exists() else 'recommendations-r3/recommendations.json' if (run/'recommendations-r3/recommendations.json').exists() else 'recommendations-r2/recommendations.json' if (run/'recommendations-r2/recommendations.json').exists() else 'recommendations/recommendations.json')]:
                            if (run/name).exists():d[key]=read(run/name)
                        if 'matrix' not in d:
                            from .recommend import scene_matrix
                            d['matrix']=scene_matrix(deck_data)
                        d['hasMockTruth']=(run/'fixtures/completion.json').is_file()
                        return self.send(d)
                    if self.path.startswith('/files/'):
                        target=(run/unquote(self.path[7:])).resolve()
                        if run not in target.parents or not target.is_file():return self.send({'error':'not found'},404)
                        return self.send(target.read_bytes(),mime=mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
                    return self.send({'error':'not found'},404)
                except (ValueError,OSError) as e:self.send({'error':str(e)},400)
            def do_POST(self):
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<48*1024*1024:raise ValueError('Request size must be below 48 MiB')
                    body=strict_json(self.rfile.read(n))
                    if not isinstance(body,dict):raise ValueError('Request must be a JSON object')
                    with lock:
                        if digest(deck_data)!=loaded_visual['boundDeckDataSha256']:raise ValueError('Loaded DeckData changed since validated startup; restart with the intended pinned dataset')
                        if self.path=='/api/scan':
                            if type(body.get('mock')) is not bool:raise ValueError('Explicit mock boolean required')
                            request=body['request']
                            native=Request(f'http://127.0.0.1:{boxport}/api/scan',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
                            t=time.perf_counter()
                            with urlopen(native,timeout=120) as r:result=json.load(r)
                            sid=uuid.uuid4().hex;dest=run/'sessions'/sid;sessions[sid]=dest
                            write(dest/'scan-response.json',result)
                            inv=inventory(result['box'],deck_data,mock=body['mock'],region=body.get('region'),recognition_dataset=loaded_visual)
                            write(dest/'inventory.json',inv)
                            return self.send({'inputBinding':{'sessionId':sid,'inventorySha256':digest(dest/'inventory.json'),'clientInputVersion':body.get('clientInputVersion'),'clientRequestId':body.get('clientRequestId')},'sessionId':sid,'scanResult':result,'inventory':inv,'elapsedMs':(time.perf_counter()-t)*1000})
                        sid=body.get('sessionId','showcase')
                        if sid not in sessions:raise ValueError('Unknown scan session')
                        dest=sessions[sid]
                        if self.path=='/api/adapt':
                            inv=read(dest/'inventory.json')
                            if body.get('completionJson') is not None:body['completion']=strict_json(body['completionJson'])
                            c=read(run/'fixtures/completion.json') if body.get('useMockTruth') else body.get('completion')
                            if body.get('useMockTruth') and not inv['mock']:raise ValueError('Mock truth cannot complete real-player inventory')
                            if body.get('useMockTruth'):
                                known={(r['kind'],r['id']) for r in inv['box']['cards']}
                                c['addMissingIdentities']=[{'kind':kind,'id':r['id'],'evidence':'Explicitly selected mock truth completion'} for kind,key in [('member','members'),('snap','snaps')] for r in c[key] if (kind,r['id']) not in known]
                            adapted=adapt(inv,c,deck_data)
                            write(dest/'reviewed-completion.json',c);write(dest/'adaptation.json',adapted);write(dest/'roster.json',adapted['roster'])
                            return self.send({'inputBinding':{'sessionId':sid,'inventorySha256':digest(dest/'inventory.json'),'rosterSha256':digest(dest/'roster.json'),'clientInputVersion':body.get('clientInputVersion'),'clientRequestId':body.get('clientRequestId')},'sessionId':sid,'adaptation':adapted,'roster':adapted['roster']})
                        if self.path=='/api/recommend':
                            if solver_bin is None or not Path(solver_bin).exists():return self.send({'error':'Final solver binary not configured'},503)
                            if not (dest/'roster.json').exists():raise ValueError('Explicit inventory completion is required before recommendation')
                            from .recommend import scene_matrix,recommend
                            matrix=read(run/'scene-matrix.json') if (run/'scene-matrix.json').exists() else scene_matrix(deck_data)
                            if body.get('requestJson') is not None:body['request']=strict_json(body['requestJson'])
                            if body.get('request') is not None:matrix={**matrix,'entries':[{'id':'custom-'+uuid.uuid4().hex[:8],'request':body['request'],'inputSource':'explicit user supplied scenario/context/seed-law',**({'requestJson':body['requestJson']} if body.get('requestJson') is not None else {})}]}
                            elif body.get('caseId'):matrix={**matrix,'entries':[e for e in matrix['entries'] if e['id']==body['caseId']]}
                            if not matrix['entries']:raise ValueError('Unknown scenario request')
                            result=recommend(solver_bin,deck_data,dest/'roster.json',matrix,dest/'recommendations'/uuid.uuid4().hex)
                            result['inputBinding']={'sessionId':sid,'inventorySha256':digest(dest/'inventory.json'),'rosterSha256':digest(dest/'roster.json'),'caseId':body.get('caseId'),'clientInputVersion':body.get('clientInputVersion'),'clientRequestId':body.get('clientRequestId')}
                            return self.send(result)
                    self.send({'error':'not found'},404)
                except IncompleteInventory as e:self.send({'error':str(e),'complete':False,'issues':e.issues},422)
                except (ValueError,KeyError,TypeError,OSError) as e:self.send({'error':str(e),'complete':False},400)
            def log_message(self,*args):pass
        print(f'Integration listening http://127.0.0.1:{port}/; BoxLens http://127.0.0.1:{boxport}/',flush=True)
        ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
    finally:
        worker.terminate();worker.wait(timeout=10);log.close()
