"""Lower the checksum-pinned BiSeNet ONNX asset without ONNX/protobuf packages.

This is a bounded reader for one fixed inference graph, not a general ONNX
runtime. Shape-only expressions are evaluated at conversion time (512x512).
"""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import numpy as np
from .face_assets import PARSING_SHA256, sha256


def varint(data,offset):
    value=0
    for shift in range(0,70,7):
        if offset>=len(data): raise ValueError('truncated protobuf integer')
        byte=data[offset];offset+=1;value|=(byte&127)<<shift
        if byte<128: return value,offset
    raise ValueError('oversized protobuf integer')


def fields(data):
    data=memoryview(data);offset=0
    while offset<len(data):
        key,offset=varint(data,offset);wire=key&7
        if not key>>3: raise ValueError('invalid protobuf field')
        if wire==0: value,offset=varint(data,offset)
        else:
            if wire==2: count,offset=varint(data,offset)
            elif wire in (1,5): count=8 if wire==1 else 4
            else: raise ValueError('unsupported protobuf wire type')
            if count>len(data)-offset: raise ValueError('truncated protobuf field')
            value=data[offset:offset+count];offset+=count
        yield key>>3,value


def repeated(rows,key):
    values=[]
    for k,value in rows:
        if k!=key:continue
        if isinstance(value,int):values.append(value)
        else:
            offset=0
            while offset<len(value):
                number,offset=varint(value,offset);values.append(number)
    return [v-(1<<64) if v>=1<<63 else v for v in values]


def tensor(data):
    rows=list(fields(data));one=dict(rows);shape=tuple(repeated(rows,1))
    if len(shape)>4 or any(x<0 or x>65536 for x in shape) or int(np.prod(shape))>16000000:
        raise ValueError('unsupported parser tensor dimensions')
    dtype={1:'<f4',6:'<i4',7:'<i8'}.get(one.get(2))
    if dtype is None: raise ValueError('unsupported parser tensor dtype')
    if 9 in one:value=np.frombuffer(one[9],dtype=dtype).copy()
    elif dtype=='<f4':
        value=np.concatenate([np.frombuffer(v,dtype='<f4') for k,v in rows if k==4]) if any(k==4 for k,v in rows) else np.empty(0,np.float32)
    else:value=np.asarray(repeated(rows,5 if dtype=='<i4' else 7),dtype=dtype)
    if value.size!=int(np.prod(shape)):raise ValueError('parser constant size mismatch')
    return bytes(one.get(8,b'')).decode(),value.reshape(shape)


def attributes(node):
    result={}
    for key,value in node:
        if key!=5:continue
        rows=list(fields(value));a=dict(rows);name=bytes(a[1]).decode()
        kind=a.get(20)
        if kind==4:result[name]=tensor(a[5])[1]
        elif kind==2:result[name]=a[3]-(1<<64) if a[3]>=1<<63 else a[3]
        elif kind==3:result[name]=bytes(a[4]).decode()
        elif kind==1:result[name]=struct.unpack('<f',a[2])[0]
        elif kind==7:result[name]=repeated(rows,8)
        else:raise ValueError('unsupported parser attribute '+name)
    return result


def lower(source):
    rows=list(fields(source));graphs=[v for k,v in rows if k==7]
    if len(graphs)!=1:raise ValueError('expected one parser graph')
    graph=list(fields(graphs[0]));constants={};shapes={};nodes=[]
    for k,value in graph:
        if k==5:
            name,array=tensor(value);constants[name]=array;shapes[name]=array.shape
    inputs=[bytes(dict(fields(v))[1]).decode() for k,v in graph if k==11]
    outputs=[bytes(dict(fields(v))[1]).decode() for k,v in graph if k==12]
    inputs=[n for n in inputs if n not in constants]
    if len(inputs)!=1 or not 1<=len(outputs)<=3:raise ValueError('unexpected parser inputs/outputs')
    outputs=outputs[:1]  # The existing parser consumes the principal prediction.
    shapes[inputs[0]]=(1,3,512,512)
    for key,value in graph:
        if key!=1:continue
        node=list(fields(value));d=dict(node);op=bytes(d[4]).decode();a=attributes(node)
        ins=[bytes(v).decode() for k,v in node if k==1];outs=[bytes(v).decode() for k,v in node if k==2]
        if len(outs)!=1:raise ValueError('unsupported parser outputs')
        out=outs[0];args=[0]*8;code=None
        if op=='Constant':constants[out]=a['value'];shapes[out]=constants[out].shape;continue
        if any(name and name not in shapes for name in ins):raise ValueError('parser graph is not topologically ordered')
        if op=='Shape':value=np.asarray(shapes[ins[0]],np.int64)
        elif op=='Gather':value=np.take(constants[ins[0]],constants[ins[1]],axis=a.get('axis',0))
        elif op=='Unsqueeze':
            axes=constants[ins[1]] if len(ins)>1 else a['axes'];value=np.expand_dims(constants[ins[0]],tuple(map(int,axes)))
        elif op=='Slice':
            source=constants[ins[0]];slices=[slice(None)]*source.ndim
            start,end=constants[ins[1]],constants[ins[2]]
            axes=constants[ins[3]] if len(ins)>3 and ins[3] else np.arange(len(start))
            steps=constants[ins[4]] if len(ins)>4 and ins[4] else np.ones(len(start),int)
            for b,e,axis,step in zip(start,end,axes,steps):slices[int(axis)]=slice(int(b),int(e),int(step))
            value=source[tuple(slices)]
        elif op=='Cast':value=constants[ins[0]].astype({1:np.float32,6:np.int32,7:np.int64}[a['to']])
        elif op=='Concat' and all(n in constants for n in ins):value=np.concatenate([constants[n] for n in ins],axis=a['axis'])
        else:
            shape=shapes[ins[0]]
            if op=='Conv':
                if a.get('group',1)!=1:raise ValueError('unsupported parser grouped convolution')
                weight=constants[ins[1]];co,ci,kh,kw=weight.shape
                sh,sw=a.get('strides',[1,1]);dh,dw=a.get('dilations',[1,1]);ph,pw,pb,pr=a.get('pads',[0]*4)
                if len(shape)!=4 or shape[1]!=ci:raise ValueError('parser convolution channel mismatch')
                oh=(shape[2]+ph+pb-dh*(kh-1)-1)//sh+1;ow=(shape[3]+pw+pr-dw*(kw-1)-1)//sw+1
                if len(ins)==2:
                    bias=out+'.zero_bias';constants[bias]=np.zeros(co,np.float32);shapes[bias]=(co,);ins.append(bias)
                shapes[out]=(1,co,oh,ow);code=100;args[:6]=[sh,sw,ph,pw,dh,dw]
            elif op in ('MaxPool','AveragePool'):
                if a.get('ceil_mode',0) or a.get('dilations',[1,1])!=[1,1]:raise ValueError('unsupported parser pooling options')
                kh,kw=a['kernel_shape'];sh,sw=a.get('strides',[1,1]);ph,pw,pb,pr=a.get('pads',[0]*4)
                shapes[out]=(shape[0],shape[1],(shape[2]+ph+pb-kh)//sh+1,(shape[3]+pw+pr-kw)//sw+1)
                code=101;args=[sh,sw,ph,pw,kh,kw,int(op=='AveragePool'),a.get('count_include_pad',0)]
            elif op=='Resize':
                if len(ins)>3 and ins[3]:size=constants[ins[3]].astype(int)
                else:size=np.floor(np.asarray(shape)*constants[ins[2]]).astype(int)
                if tuple(size[:2])!=shape[:2]:raise ValueError('parser resize changes channels')
                shapes[out]=tuple(map(int,size));code=102
                args[:3]=[{'nearest':0,'linear':1}[a.get('mode','nearest')],
                          {'half_pixel':0,'align_corners':1,'asymmetric':2}[a.get('coordinate_transformation_mode','half_pixel')],
                          {'floor':0,'round_prefer_floor':1,'round_prefer_ceil':2,'ceil':3}[a.get('nearest_mode','round_prefer_floor')]]
                ins=ins[:1]
            elif op=='Concat':
                axis=a['axis'];size=list(shape);size[axis]=sum(shapes[n][axis] for n in ins);shapes[out]=tuple(size);code=2;args[0]=axis
            elif op in ('Add','Mul'):
                shapes[out]=np.broadcast_shapes(*[shapes[n] for n in ins]);code={'Add':0,'Mul':18}[op]
            elif op in ('Relu','Sigmoid'):shapes[out]=shape;code={'Relu':19,'Sigmoid':14}[op]
            else:raise ValueError('unsupported parser operator '+op)
            nodes.append((code,ins,out,args));continue
        constants[out]=np.asarray(value);shapes[out]=constants[out].shape
    used=set(inputs+outputs);live=[]
    for entry in reversed(nodes):
        _,ins,out,_=entry
        if out in used:used.update(ins);live.append(entry)
    nodes=list(reversed(live))
    names=sorted(used);indices={n:i for i,n in enumerate(names)}
    tensors=[(shapes[n],None if n not in constants else np.ascontiguousarray(constants[n],dtype='<f4')) for n in names]
    for shape,_ in tensors:
        if len(shape)>4 or any(x<1 for x in shape):raise ValueError('invalid lowered parser tensor')
    return tensors,[(c,[indices[n] for n in ins],indices[out],a) for c,ins,out,a in nodes],indices[inputs[0]],indices[outputs[0]]


def export(source,output):
    source=Path(source);output=Path(output)
    if sha256(source)!=PARSING_SHA256:raise ValueError('verified face parsing weights required')
    tensors,nodes,inp,out=lower(source.read_bytes());output.parent.mkdir(parents=True,exist_ok=True)
    ints=lambda values:struct.pack('<'+'i'*len(values),*values)
    with tempfile.TemporaryDirectory(prefix='.parser-',dir=output.parent) as stage:
        path=Path(stage)/output.name
        with path.open('wb') as f:
            f.write(b'VHFACE2\0');f.write(ints([len(tensors),len(nodes),1,1,inp,out]))
            for shape,value in tensors:
                f.write(ints([len(shape),*shape,*([1]*(4-len(shape))),0 if value is None else value.size]))
                if value is not None:f.write(value.tobytes())
            for code,ins,out,args in nodes:f.write(ints([code,len(ins),out,*args,*ins]))
        digest=sha256(path)
        receipt=dict(format='vhuman.native_parsing.v1',source_sha256=PARSING_SHA256,sha256=digest,operators=len(nodes))
        meta=Path(stage)/(output.name+'.json');meta.write_text(json.dumps(receipt,indent=2)+'\n')
        path.replace(output);meta.replace(output.with_name(output.name+'.json'))
    return receipt
