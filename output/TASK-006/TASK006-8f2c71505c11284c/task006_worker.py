
# Original baseline structure: Dataset -> Collator -> DataLoader -> AdamW training.
import argparse, csv, gc, hashlib, json, math, os, random, re, sys, time, traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any


def write_json(path,obj):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8'); tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def block_network():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def guard(event,args):
        if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}: raise RuntimeError('로컬 실행 중 네트워크 접근 금지')
    sys.addaudithook(guard)


def read_table(path,columns):
    import pandas as pd
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if set(columns)-set(df.columns): raise ValueError('필수 컬럼 누락')
    if df.empty or df.id.duplicated().any(): raise ValueError('빈 데이터 또는 중복 ID')
    for c in columns:
        if df[c].str.strip().eq('').any(): raise ValueError(f'빈 필수 항목: {c}')
    return df


def resolve_image(root,value):
    value=str(value).replace('\\','/')
    root=Path(root).resolve(); p=(root/value).resolve()
    if '://' in value or ':' in value or not p.is_relative_to(root) or not p.is_file(): raise FileNotFoundError(p)
    return p


SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)


def build_mc_prompt(question,a,b,c,d):
    return (f'{question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n'
            '정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.')


def parse_answer(text):
    # Baseline's implicit 'a' replaced by explicit failure. No fallback in this baseline.
    s=re.sub(r'^(?:answer|정답)\s*[:：]\s*','',str(text).strip(),flags=re.I)
    m=re.fullmatch(r'(?:\(([a-d])\)|([a-d]))[.。]?',s,flags=re.I)
    return (m.group(1) or m.group(2)).lower() if m else None






from torch.utils.data import Dataset


class VQAMCDataset(Dataset):
    def __init__(self,df,processor,train=True,data_dir=None):
        self.df=df.reset_index(drop=True); self.processor=processor; self.train=train; self.data_dir=data_dir
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        from PIL import Image,ImageOps
        row=self.df.iloc[i]
        with Image.open(resolve_image(self.data_dir,row['path'])) as f:
            img=ImageOps.exif_transpose(f).convert('RGB')
        user_text=build_mc_prompt(*(str(row[k]) for k in ['question','a','b','c','d']))
        messages=[{'role':'system','content':[{'type':'text','text':SYSTEM_INSTRUCT}]},
                  {'role':'user','content':[{'type':'image','image':img},{'type':'text','text':user_text}]}]
        if self.train:
            messages.append({'role':'assistant','content':[{'type':'text','text':str(row['answer'])}]})
        return {'messages':messages,'image':img}


@dataclass
class DataCollator:
    processor: Any
    train: bool=True
    max_input_tokens: int=4096
    def __call__(self,batch):
        texts,images=[],[]
        for sample in batch:
            texts.append(self.processor.apply_chat_template(sample['messages'],tokenize=False,
                add_generation_prompt=not self.train,enable_thinking=False))
            images.append(sample['image'])
        enc=self.processor(text=texts,images=images,padding=True,return_tensors='pt',add_special_tokens=False)
        if enc['input_ids'].shape[1]>self.max_input_tokens: raise ValueError('입력 길이 초과: 자동 잘라내기 없음')
        if self.train:
            # Original baseline full-sequence labels retained, including media/role tokens.
            # Only artificial padding is ignored. Batch size 1 normally has no padding.
            enc['labels']=enc['input_ids'].clone()
            enc['labels'][enc['attention_mask']==0]=-100
        return enc






class ModelAdapter:
    """Stage runner bridge; actual input/training uses baseline Dataset and Collator."""
    def __init__(self,cfg,out):
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText,BitsAndBytesConfig
        self.cfg=cfg;self.out=out; self.dtype=torch.bfloat16;self.device='cuda:0'
        if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 지원 GPU 필요')
        self.processor=AutoProcessor.from_pretrained(cfg['model_dir'],local_files_only=True)
        ip=self.processor.image_processor; pixels=cfg['pixel_budget']
        ip.size={'shortest_edge':pixels,'longest_edge':pixels}
        if hasattr(ip,'min_pixels'): ip.min_pixels=pixels
        if hasattr(ip,'max_pixels'): ip.max_pixels=pixels
        self.tokenizer=self.processor.tokenizer
        bnb_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=self.dtype)
        self.model=AutoModelForImageTextToText.from_pretrained(cfg['model_dir'],
            local_files_only=True,quantization_config=bnb_config,device_map={'':'cuda:0'},
            dtype=self.dtype,attn_implementation='sdpa')
        self.model.config.use_cache=False; self.model.eval()
        self.processor.save_pretrained(out/'processor')
        write_json(out/'model_config.json',self.model.config.to_dict())
        write_json(out/'processor_settings.json',ip.to_dict())
    def add_lora(self):
        import torch
        from peft import LoraConfig,get_peft_model
        # Memory-safe k-bit preparation: avoid full embedding FP32 expansion on 16 GB.
        for name,p in self.model.named_parameters():
            p.requires_grad_(False)
            if 'norm' in name.lower() and p.ndim==1 and p.is_floating_point(): p.data=p.data.to(torch.float32)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.enable_input_require_grads()
        lora_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            task_type='CAUSAL_LM')
        self.model=get_peft_model(self.model,lora_config)
        self.model.print_trainable_parameters()
        write_json(self.out/'lora_parameters.json',[{'name':n,'shape':list(p.shape)} for n,p in self.model.named_parameters() if p.requires_grad])
    def move(self,batch):
        return {k:v.to(self.device,dtype=self.dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}
    def encode(self,row,training=False):
        import pandas as pd
        ds=VQAMCDataset(pd.DataFrame([row]),self.processor,training,self.cfg['data_dir'])
        sample=ds[0]; enc=DataCollator(self.processor,training,self.cfg['max_input_tokens'])([sample])
        grid=enc.get('image_grid_thw'); gs=grid.tolist() if grid is not None else []
        ip=self.processor.image_processor; patch=getattr(ip,'patch_size',16); merge=getattr(ip,'merge_size',2)
        self.last_image_meta={'original_size':list(sample['image'].size),'grid_thw':gs,
            'processed_hw':[[int(g[1])*patch,int(g[2])*patch] for g in gs],
            'visual_tokens':sum(math.prod(g)//(merge*merge) for g in gs)}
        return self.move(enc)
    def loss(self,inputs): return self.model(**inputs,use_cache=False).loss
    def generate(self,inputs):
        import torch
        self.model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=self.dtype):
            ids=self.model.generate(**inputs,max_new_tokens=self.cfg['max_new_tokens'],do_sample=False,
                use_cache=True,eos_token_id=self.tokenizer.eos_token_id)
        # Original baseline decoded the whole input; now decode generated tokens only.
        return self.tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    def reload_adapter(self,path):
        from peft import PeftModel
        base=self.model.unload()
        self.model=PeftModel.from_pretrained(base,str(path),local_files_only=True,is_trainable=False)
        self.model.eval()






def evaluate(adapter,df,label,out):
    import torch,pandas as pd
    adapter.model.eval();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    rows=[]
    fields=['id','answer','raw_output','parse_failed','fallback_used','gold','correct','strict_correct','group_id','question_type','image_meta']
    with open(out/f'{label}_predictions.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(df.to_dict('records')):
            inputs=adapter.encode({k:v for k,v in row.items() if k!='answer'})
            raw=adapter.generate(inputs);answer=parse_answer(raw)
            rec={'id':row['id'],'answer':answer or '', 'raw_output':raw,'parse_failed':answer is None,
                 'fallback_used':False,'gold':row['answer'],'correct':answer==row['answer'],
                 'strict_correct':answer==row['answer'],'group_id':row['group_id'],'question_type':row['question_type'],
                 'image_meta':json.dumps(adapter.last_image_meta)}
            writer.writerow(rec);f.flush();rows.append(rec);del inputs
            if (i+1)%25==0 or i+1==len(df): print(label,i+1,'/',len(df),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-start;n=len(rows)
    metrics={'n':n,'correct_n':sum(r['correct'] for r in rows),'accuracy':sum(r['correct'] for r in rows)/n,
        'parse_failure_rate':sum(r['parse_failed'] for r in rows)/n,'fallback_usage_rate':0.,
        'seconds':seconds,'seconds_per_sample':seconds/n,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    write_json(out/f'{label}_metrics.json',metrics);print(metrics,flush=True)
    return rows,metrics






def train_one_epoch(adapter,df,cfg,out):
    import torch,pandas as pd
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    # Baseline DataLoader, AdamW, warmup, accumulation, one epoch retained.
    train_ds=VQAMCDataset(df,adapter.processor,train=True,data_dir=cfg['data_dir'])
    train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,
        generator=torch.Generator().manual_seed(cfg['seed']),
        collate_fn=DataCollator(adapter.processor,True,cfg['max_input_tokens']),num_workers=0)
    model=adapter.model; GRAD_ACCUM=cfg['gradient_accumulation']
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'])
    num_training_steps=math.ceil(len(train_loader)/GRAD_ACCUM)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(num_training_steps*.03),num_training_steps)
    # BF16 needs no GradScaler; original mixed FP16 compute/BF16 autocast is unified.
    model.train(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();rows=[];running=0.
    for step,batch in enumerate(train_loader,start=1):
        batch=adapter.move(batch)
        # Correct denominator and optimizer step for the final incomplete group.
        group_start=((step-1)//GRAD_ACCUM)*GRAD_ACCUM
        group_size=min(GRAD_ACCUM,len(train_loader)-group_start)
        with torch.autocast('cuda',dtype=adapter.dtype):
            outputs=model(**batch,use_cache=False);loss=outputs.loss
        if not torch.isfinite(loss): raise FloatingPointError('loss NaN/Inf')
        raw_loss=float(loss.detach());(loss/group_size).backward();running+=raw_loss
        del outputs,loss,batch
        if step%GRAD_ACCUM==0 or step==len(train_loader):
            grads=[p.grad for p in params if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads): raise FloatingPointError('gradient 오류')
            optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step()
            rec={'update':len(rows)+1,'mean_loss':running/group_size,'lr':scheduler.get_last_lr()[0]};rows.append(rec);running=0.
            with open(out/'train_updates.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
            print('train',rec['update'],'/',num_training_steps,rec,flush=True)
    torch.cuda.synchronize()
    metrics={'train_n':len(df),'epochs':1,'updates':len(rows),'seconds':time.perf_counter()-started,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    pd.DataFrame(rows).to_csv(out/'train_log.csv',index=False);write_json(out/'train_metrics.json',metrics)
    del optimizer,scheduler,params,grads,train_loader;gc.collect();torch.cuda.empty_cache()
    return metrics




def question_type(text):
    # Heuristic diagnostic tags, not verified semantic labels.
    if re.search(r'아닌|않은|않는|없는|제외', text): return 'negation'
    if re.search(r'가격|얼마|원인가|금액|할인', text): return 'price'
    if re.search(r'왼쪽|오른쪽|위쪽|아래|옆|위치', text): return 'position'
    if re.search(r'몇|숫자|날짜|시간|번호|수량', text): return 'number'
    return 'text_other'


def take_groups(frame, target, seed):
    import numpy as np
    # Never split a group. Greedy size selection over repeated seeded permutations;
    # use answer/type proportions to break ties and minimize distribution shift.
    groups = frame.groupby('group_id', sort=True).size()
    if target >= len(frame): return set(groups.index)
    rng = np.random.default_rng(seed)
    best, best_score = None, float('inf')
    answer_ref = frame.answer.value_counts(normalize=True)
    type_ref = frame.question_type.value_counts(normalize=True)
    for _ in range(100):
        chosen, n = [], 0
        for g in rng.permutation(groups.index.to_numpy()):
            size = int(groups[g])
            if abs(n + size - target) < abs(n-target):
                chosen.append(g); n += size
        sub = frame[frame.group_id.isin(chosen)]
        if sub.empty: continue
        def distance(col, ref):
            return (sub[col].value_counts(normalize=True).reindex(ref.index,fill_value=0)-ref).abs().sum()
        score = abs(n-target) + .1 * (distance('answer',answer_ref)+distance('question_type',type_ref))
        if score < best_score: best, best_score = set(chosen), score
    if best is None: raise ValueError('그룹 크기로 인해 분할 불가')
    return best


def audit(cfg, out):
    import pandas as pd
    import numpy as np
    from PIL import Image, ImageOps
    root = Path(cfg['data_dir'])
    df = read_table(root/'train.csv', ['id','path','question','a','b','c','d','answer']).sort_values('id').reset_index(drop=True)
    if not df.answer.isin(list('abcd')).all(): raise ValueError('라벨은 소문자 a~d여야 합니다.')
    findings = []
    for row in df.to_dict('records'):
        for col in ['question','a','b','c','d']:
            if row[col] != row[col].strip(): findings.append({'id':row['id'],'field':col,'reason':'leading/trailing whitespace','action':'report only'})
            if '\ufffd' in row[col]: findings.append({'id':row['id'],'field':col,'reason':'replacement character','action':'review'})
        if len({row[c].strip() for c in 'abcd'}) < 4:
            findings.append({'id':row['id'],'field':'options','reason':'duplicate options','action':'review'})
    pd.DataFrame(findings,columns=['id','field','reason','action']).to_csv(out/'data_findings.csv',index=False)
    images, errors = [], []
    for i,row in enumerate(df.to_dict('records')):
        try:
            p = resolve_image(root,row['path'])
            with Image.open(p) as raw:
                raw.load()
                orientation = raw.getexif().get(274,1)
                im = ImageOps.exif_transpose(raw).convert('RGB')
                pixel_hash = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                tiny=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS))
                bits=(tiny[:,1:]>tiny[:,:-1]).reshape(-1)
                dh=sum(int(v)<<j for j,v in enumerate(bits))
                stat=p.stat()
                images.append(dict(id=row['id'],path=row['path'],width=im.width,height=im.height,
                    orientation=orientation,pixel_sha256=pixel_hash,dhash=f'{dh:016x}',
                    file_sha256=sha256(p),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
        except Exception as exc: errors.append(dict(id=row['id'],path=row['path'],error=str(exc)))
        if (i+1)%250==0: print('image audit',i+1,'/',len(df),flush=True)
    pd.DataFrame(errors,columns=['id','path','error']).to_csv(out/'image_errors.csv',index=False)
    if errors: raise RuntimeError(f'누락/손상 이미지 {len(errors)}개. image_errors.csv 확인. 자동 제외하지 않습니다.')
    parent=list(range(len(images)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parent[max(a,b)]=min(a,b)
    exact={}; candidates=[]; hashes=[int(x['dhash'],16) for x in images]
    for i,row in enumerate(images):
        h=row['pixel_sha256']
        if h in exact: union(i,exact[h])
        else: exact[h]=i
        ratio=row['width']/row['height']
        for j in range(i):
            if images[j]['pixel_sha256']==h: continue
            distance=(hashes[i]^hashes[j]).bit_count()
            if distance<=cfg['near_hash_distance'] and abs(math.log(ratio/(images[j]['width']/images[j]['height'])))<=.05:
                # Conservative candidate grouping; false positives can reduce train pool.
                union(i,j)
                candidates.append({'id_a':images[j]['id'],'id_b':row['id'],'distance':distance,
                                   'policy':'same group conservatively; not human verified'})
    df['group_id']=['g_'+images[find(i)]['id'] for i in range(len(images))]
    df['question_type']=df.question.map(question_type)
    for i,row in enumerate(images): row['group_id']=df.iloc[i].group_id
    pd.DataFrame(images).to_csv(out/'image_audit.csv',index=False)
    pd.DataFrame(candidates,columns=['id_a','id_b','distance','policy']).to_csv(out/'near_duplicate_candidates.csv',index=False)
    if cfg['train_n']+cfg['valid_n']+cfg['holdout_n']>=len(df):
        raise ValueError('데이터 개수보다 분할 요청이 큽니다. config 값을 확인하세요.')
    final=take_groups(df,cfg['holdout_n'],cfg['seed'])
    remaining=df[~df.group_id.isin(final)]
    valid=take_groups(remaining,cfg['valid_n'],cfg['seed']+1)
    remaining=remaining[~remaining.group_id.isin(valid)]
    train=take_groups(remaining,cfg['train_n'],cfg['seed']+2)
    df['split']=['holdout' if g in final else 'valid' if g in valid else 'train' if g in train else 'pool' for g in df.group_id]
    if not set(['train','valid','holdout']).issubset(set(df.split)): raise ValueError('빈 분할이 있습니다.')
    assert df.groupby('group_id').split.nunique().max()==1
    split=df[['id','split','group_id','question_type']]
    split.to_csv(out/'split_manifest.csv',index=False)
    for name in ['train','valid','holdout','pool']:
        split[split.split==name].to_csv(out/f'{name}_ids.csv',index=False)
    dist=pd.crosstab(df.split,df.answer).reindex(columns=list('abcd'),fill_value=0)
    dist.to_csv(out/'answer_distribution.csv')
    pd.crosstab(df.split,df.question_type).to_csv(out/'type_distribution.csv')
    manifest={'csv_sha256':sha256(root/'train.csv'),'split_sha256':sha256(out/'split_manifest.csv'),
              'image_audit_sha256':sha256(out/'image_audit.csv'),'counts':df.split.value_counts().to_dict(),
              'groups':int(df.group_id.nunique()),'near_candidate_pairs':len(candidates),
              'near_policy':'dHash candidate transitive grouping; human review pending; misses possible',
              'question_type_policy':'regex heuristic only','review_02':'pending','dev_used':False,
              'test_used':False,'findings':len(findings)}
    write_json(out/'data_manifest.json',manifest)
    print(dist.to_string(),flush=True); print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
    return manifest


def completed(root, stage):
    p=root/f'{stage}_complete.json'
    if not p.exists(): return None
    rec=json.loads(p.read_text(encoding='utf-8'))
    for name,digest in rec['artifacts'].items():
        path=Path(rec['directory'])/name
        if not path.is_file() or sha256(path)!=digest: raise RuntimeError(f'완료 결과가 수정/누락되었습니다: {path}')
    return rec


def load_split(cfg, root):
    import pandas as pd
    record=completed(root,'audit')
    if record is None: raise RuntimeError('audit 단계를 먼저 실행하세요.')
    path=Path(record['directory'])
    meta=json.loads((path/'data_manifest.json').read_text(encoding='utf-8'))
    if sha256(Path(cfg['data_dir'])/'train.csv')!=meta['csv_sha256']: raise RuntimeError('train.csv 변경: 새 SESSION_TAG로 분할부터 실행하세요.')
    # Validate image bytes: changed data cannot silently reuse completed experiments.
    images=pd.read_csv(path/'image_audit.csv',dtype=str)
    for r in images.to_dict('records'):
        p=resolve_image(cfg['data_dir'],r['path'])
        if sha256(p)!=r['file_sha256']: raise RuntimeError(f"이미지 변경: {r['id']}. 새 SESSION_TAG 필요")
    df=pd.read_csv(Path(cfg['data_dir'])/'train.csv',dtype=str,keep_default_na=False)
    split=pd.read_csv(path/'split_manifest.csv',dtype=str)
    merged=df.merge(split,on='id',validate='one_to_one')
    assert len(merged)==len(df) and merged.groupby('group_id').split.nunique().max()==1
    return merged[merged.split=='train'].copy(),merged[merged.split=='valid'].copy(),meta


def gpu_environment(cfg,out):
    import torch, numpy as np, platform
    from importlib.metadata import version
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU를 찾을 수 없습니다. 드라이버/설치를 확인하세요.')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16)
    assert float((x@x)[0,0])==32.; del x
    info={'python':sys.version,'os':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name(),'vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
          'packages':{p:version(p) for p in ['transformers','peft','bitsandbytes','accelerate']},
          'network':'blocked; local model files only'}
    write_json(out/'environment.json',info); print(info,flush=True)


def smoke(adapter, tr, cfg, out):
    import torch
    row=tr.iloc[0].to_dict()
    clean={k:v for k,v in row.items() if k!='answer'}
    before=adapter.generate(adapter.encode(clean))
    adapter.add_lora(); adapter.model.train(); torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); inputs=adapter.encode(row,training=True)
    if 'pixel_values' not in inputs or inputs['pixel_values'].numel()==0: raise RuntimeError('이미지 입력이 없습니다.')
    ids=inputs['input_ids'][0].tolist(); labels=inputs['labels'][0].tolist()
    import pandas as pd
    pd.DataFrame([{'position':i,'token_id':t,'token':adapter.tokenizer.convert_ids_to_tokens(t),
                   'supervised':labels[i]!=-100} for i,t in enumerate(ids)]).to_csv(out/'supervised_tokens.csv',index=False)
    assert sum(x!=-100 for x in labels)>0
    with torch.autocast('cuda',dtype=adapter.dtype): loss=adapter.loss(inputs)
    if not torch.isfinite(loss): raise FloatingPointError('유한하지 않은 loss')
    loss.backward()
    grads=[p.grad for p in adapter.model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads) and any(bool(g.abs().max()>0) for g in grads)
    torch.cuda.synchronize()
    result={'id':row['id'],'loss':float(loss.detach()),'generation':before,'parsed':parse_answer(before),
            'finite_nonzero_gradients':True,'optimizer_step_performed':False,
            'seconds_one_forward_backward':time.perf_counter()-start,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'supervised_tokens':sum(x!=-100 for x in labels),'image_meta':adapter.last_image_meta,
            'note':'smoke only; runtime estimate is rough, not measured epoch duration'}
    adapter.model.zero_grad(set_to_none=True); write_json(out/'smoke_metrics.json',result)
    print(result,flush=True)
    return result


def compare(root, out):
    import pandas as pd
    b=completed(root,'base'); l=completed(root,'lora_eval')
    if b is None or l is None: raise RuntimeError('base와 lora_eval을 먼저 완료하세요.')
    a=pd.read_csv(Path(b['directory'])/'valid_base_predictions.csv',keep_default_na=False)
    z=pd.read_csv(Path(l['directory'])/'valid_lora_predictions.csv',keep_default_na=False)
    m=a.merge(z,on='id',suffixes=('_base','_lora'),validate='one_to_one')
    assert len(m)==len(a)==len(z) and (m.gold_base==m.gold_lora).all()
    good_a=m.answer_base==m.gold_base; good_z=m.answer_lora==m.gold_lora
    m['transition']=['gain' if not x and y else 'loss' if x and not y else 'same_correct' if x else 'same_wrong' for x,y in zip(good_a,good_z)]
    m.to_csv(out/'paired_predictions.csv',index=False)
    m[m.transition.isin(['gain','loss'])].to_csv(out/'changed_answers.csv',index=False)
    errors=m[~good_z].copy(); errors['manual_error_category']=''; errors['review_note']=''
    errors.to_csv(out/'error_review_template.csv',index=False)
    rows=[]
    for label,rec in [('base',b),('lora',l)]:
        metrics=json.loads((Path(rec['directory'])/f'valid_{label}_metrics.json').read_text())
        rows.append({'model':label,**metrics})
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    types=m.groupby('question_type_base').apply(lambda g:pd.Series({'n':len(g),
        'base_accuracy':(g.answer_base==g.gold_base).mean(),'lora_accuracy':(g.answer_lora==g.gold_lora).mean()}),include_groups=False)
    types.to_csv(out/'type_comparison.csv')
    result={'gain':int((~good_a & good_z).sum()),'loss':int((good_a & ~good_z).sum()),
            'accuracy_delta':float(good_z.mean()-good_a.mean()),'n':len(m),
            'adoption':'pending; no automatic model selection','review_02':'pending',
            'holdout_evaluated':False,'test_evaluated':False}
    write_json(out/'summary.json',result)
    (out/'PROJECT_STATUS_update.md').write_text('# TASK-006-R1 반영 후보\n\n기준 평가 실행 완료. 독립 검토 미완료.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n실제 파일 위치: '+str(root),encoding='utf-8')
    (out/'CHANGELOG.md').write_text('# TASK-006-R1 실행 기록\n\n고정 분할에서 base/새 LoRA 평가 완료. 원본/최고 모델/공식 PROJECT_STATUS 수정 없음. 채택·제출·02 검토 미완료.\n',encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print(result,flush=True)
    return result


def main(cfg,stage):
    block_network()
    root=Path(cfg['run_dir']); root.mkdir(parents=True,exist_ok=True)
    # A fresh process per stage frees all CUDA memory when it exits.
    if stage!='audit': tr,va,data_info=load_split(cfg,root)
    old=completed(root,stage)
    if old:
        print('완료 단계 재사용:',stage,old['directory'],flush=True);return
    import uuid
    out=root/(stage+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);out.mkdir()
    write_json(out/'config.json',cfg)
    write_json(root/f'{stage}_status.json',{'state':'running','directory':str(out)})
    try:
        if stage=='audit': result=audit(cfg,out)
        elif stage=='compare': result=compare(root,out)
        else:
            if stage in ['base','train','lora_eval'] and completed(root,'smoke') is None:
                raise RuntimeError('smoke 검사를 먼저 완료하세요.')
            if stage=='lora_eval' and completed(root,'train') is None: raise RuntimeError('train 먼저 실행 필요')
            gpu_environment(cfg,out)
            adapter=ModelAdapter(cfg,out)
            if stage=='smoke': result=smoke(adapter,tr,cfg,out)
            elif stage=='base': _,result=evaluate(adapter,va,'valid_base',out)
            elif stage=='train':
                adapter.add_lora()
                result=train_one_epoch(adapter,tr,cfg,out)
                checkpoint=out/'adapter_epoch1'; adapter.model.save_pretrained(checkpoint);adapter.processor.save_pretrained(checkpoint)
                # Compare same few validation predictions immediately before/after disk reload.
                probe=va.head(min(5,len(va)))
                _,p=evaluate(adapter,probe,'reload_before',out)
                adapter.reload_adapter(checkpoint)
                _,q=evaluate(adapter,probe,'reload_after',out)
                import pandas as pd
                a=pd.read_csv(out/'reload_before_predictions.csv',keep_default_na=False)
                b=pd.read_csv(out/'reload_after_predictions.csv',keep_default_na=False)
                same=a[['id','raw_output','answer']].equals(b[['id','raw_output','answer']])
                write_json(out/'reload_check.json',{'n':len(probe),'identical_outputs':same})
                if not same: raise RuntimeError('저장 전/후 예측 불일치. 로그를 검토하세요.')
                result['checkpoint']=str(checkpoint)
            elif stage=='lora_eval':
                from peft import PeftModel
                train_rec=completed(root,'train'); checkpoint=Path(train_rec['directory'])/'adapter_epoch1'
                adapter.model=PeftModel.from_pretrained(adapter.model,str(checkpoint),local_files_only=True,is_trainable=False)
                adapter.base=adapter.model.get_base_model()
                _,result=evaluate(adapter,va,'valid_lora',out)
            else: raise ValueError(stage)
        files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}
        rec={'stage':stage,'directory':str(out),'result':result,'artifacts':files}
        write_json(root/f'{stage}_complete.json',rec)
        write_json(root/f'{stage}_status.json',{'state':'completed','directory':str(out)})
    except BaseException as exc:
        write_json(root/f'{stage}_status.json',{'state':'failed_or_interrupted','directory':str(out),'error':str(exc)})
        raise


if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    main(cfg,sys.argv[2])

