"""Optional BoxLens workflow using the existing ournotes-deck contracts."""
import json
from pathlib import Path
from .common import read, write, digest, strict_json
from .adapter import inventory, adapt, IncompleteInventory


def add_parsers(sub):
    def inputs(command, help):
        parser=sub.add_parser(command,help=help)
        parser.add_argument('--data',type=Path,required=True)
        parser.add_argument('--deck-data',type=Path,required=True)
        return parser
    p=inputs('deck-ui','Local screenshot review and ournotes-deck recommendation UI')
    p.add_argument('--solver-bin',type=Path,required=True)
    p.add_argument('--workspace','--run',dest='workspace',type=Path,required=True)
    p.add_argument('--port',type=int,default=18790)
    p.add_argument('--box-port',type=int,default=18793)
    p=inputs('deck-adapt','Review bdon-box/1 and export the existing Roster schema')
    p.add_argument('--box',type=Path,required=True)
    p.add_argument('--completion',type=Path)
    p.add_argument('--region',required=True)
    p.add_argument('--mock',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p=inputs('deck-mock','Generate declared mock screenshots using the original renderer')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--members',type=int,default=7)
    p.add_argument('--snaps',type=int,default=3)
    p=sub.add_parser('deck-recommend',help='Run one exact request through the real solver')
    p.add_argument('--deck-data',type=Path,required=True)
    p.add_argument('--solver-bin',type=Path,required=True)
    p.add_argument('--roster',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)


def run(args):
    if args.command=='deck-ui':
        from .server import serve
        if not args.solver_bin.is_file():raise ValueError('Solver executable not found')
        serve(args.workspace,args.port,args.solver_bin.resolve(),args.box_port,
              data=args.data,deck_data=args.deck_data)
    elif args.command=='deck-mock':
        from .visual_data import binding
        from .mock import generate
        binding(args.data,args.deck_data)
        if args.output.exists():raise ValueError('Mock output must be a new directory')
        manifest=generate(args.data,args.deck_data,args.output,args.members,args.snaps,newest_snaps=True)
        print(json.dumps({'source':'mock','screenshots':len(manifest['screenshots'])}))
    elif args.command=='deck-adapt':
        from .visual_data import binding
        inv=inventory(read(args.box),args.deck_data,mock=args.mock,region=args.region,
                      recognition_dataset=binding(args.data,args.deck_data))
        write(args.output/'inventory.json',inv)
        try:
            result=adapt(inv,read(args.completion) if args.completion else None,args.deck_data)
        except IncompleteInventory as error:
            write(args.output/'needs-review.json',{'complete':False,'issues':error.issues})
            print(str(error))
            return 2
        write(args.output/'adaptation.json',result)
        write(args.output/'roster.json',result['roster'])
        print(json.dumps({'complete':True,'mock':inv['mock'],'corrections':len(result['corrections'])}))
    elif args.command=='deck-recommend':
        from .recommend import recommend
        raw=args.request.read_bytes().decode('utf-8')
        request=strict_json(raw)
        matrix={'deckDataSha256':digest(args.deck_data),
                'entries':[{'id':'user-request','request':request,'requestJson':raw}]}
        result=recommend(args.solver_bin.resolve(),args.deck_data,args.roster,matrix,args.output)
        print(json.dumps(result,ensure_ascii=False))
        return result['cases'][0].get('exitCode',2)
    return 0
