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
    if cfg['valid_n']+cfg['holdout_n']>=len(df):
        raise ValueError('데이터 개수보다 분할 요청이 큽니다. config 값을 확인하세요.')
    final=take_groups(df,cfg['holdout_n'],cfg['seed'])
    remaining=df[~df.group_id.isin(final)]
    valid=take_groups(remaining,cfg['valid_n'],cfg['seed']+1)
    remaining=remaining[~remaining.group_id.isin(valid)]
    train=set(remaining.group_id)  # Every group outside valid and holdout.
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


def answer_span(full,empty,eos):
    start=0
    while start<min(len(full),len(empty)) and full[start]==empty[start]:start+=1
    suffix=0
    while suffix<min(len(full)-start,len(empty)-start) and full[-1-suffix]==empty[-1-suffix]:suffix+=1
    end=len(full)-suffix
    if start>=end:raise RuntimeError('정답 token span을 찾지 못했습니다.')
    if end>=len(full) or full[end]!=eos:
        raise RuntimeError('정답 바로 뒤 종료 토큰이 예상과 다릅니다. 템플릿 검토 필요; 자동 우회 없음.')
    return start,end


def install_answer_collator(module):
    import copy
    original=module.DataCollator
    class AnswerOnlyCollator(original):
        def __call__(self,batch):
            enc=super().__call__(batch)
            if not self.train:return enc
            if len(batch)!=1:raise ValueError('이 실험은 baseline과 동일하게 batch_size=1')
            empty=[]
            for sample in batch:
                msgs=copy.deepcopy(sample['messages'])
                if msgs[-1]['role']!='assistant':raise RuntimeError('assistant 정답 없음')
                gold=msgs[-1]['content'][0]['text']
                msgs[-1]['content']=[{'type':'text','text':''}]
                empty.append({'messages':msgs,'image':sample['image']})
            empty_enc=super().__call__(empty)
            full=enc['input_ids'][0].tolist();blank=empty_enc['input_ids'][0].tolist()
            start,end=answer_span(full,blank,self.processor.tokenizer.eos_token_id)
            decoded=self.processor.tokenizer.decode(full[start:end],skip_special_tokens=False).strip()
            if decoded!=gold:raise RuntimeError(f'정답 토큰 검증 실패: {decoded!r} != {gold!r}')
            enc['labels'].fill_(-100)
            enc['labels'][0,start:end+1]=enc['input_ids'][0,start:end+1]
            return enc
    module.DataCollator=AnswerOnlyCollator


def audit_targets(module,adapter,train,out):
    import pandas as pd
    rows=[]
    for row in train.head(3).to_dict('records'):
        enc=adapter.encode(row,training=True)
        ids=enc['input_ids'][0].tolist();labels=enc['labels'][0].tolist()
        for i,t in enumerate(ids):
            rows.append({'id':row['id'],'position':i,'token_id':t,
                         'token':adapter.tokenizer.convert_ids_to_tokens(t),'supervised':labels[i]!=-100})
        del enc
    pd.DataFrame(rows).to_csv(out/'supervised_tokens.csv',index=False,encoding='utf-8-sig')



import os,sys,json,csv,hashlib,importlib.util,time,uuid,contextlib
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def tables(cfg):
    import pandas as pd
    test=pd.read_csv(cfg['test_csv'],dtype=str,keep_default_na=False)
    sample=pd.read_csv(cfg['sample_csv'],dtype=str,keep_default_na=False)
    cols=['id','path','question','a','b','c','d']
    if not set(cols)<=set(test.columns) or list(sample.columns)!=['id','answer']:raise ValueError('test/sample_submission 컬럼 확인 필요')
    if test.empty or len(test)!=len(sample):raise ValueError('test/sample 행 수 불일치')
    for name,d in [('test',test),('sample',sample)]:
        if d.id.duplicated().any() or d.id.str.strip().eq('').any():raise ValueError(f'{name} ID 중복/누락')
    if set(test.id)!=set(sample.id):raise ValueError('test/sample ID 집합 불일치')
    for c in cols:
        if test[c].str.strip().eq('').any():raise ValueError(f'test 빈 값: {c}')
    # Sample IDs are authoritative ordering; no question/choice text modifications.
    test=test.set_index('id').loc[sample.id].reset_index()[cols]
    return test,sample


def load_progress(root,fingerprint,ids):
    p=root/'progress.json'
    progress=read(p) if p.exists() else {'fingerprint':fingerprint,'chunks':[]}
    if progress['fingerprint']!=fingerprint:raise RuntimeError('재개 설정 불일치')
    records=[]
    for chunk in progress['chunks']:
        path=root/chunk['file']
        if sha(path)!=chunk['sha256']:raise RuntimeError('부분 예측 파일 변경')
        records.extend(read(path))
    if [r['id'] for r in records]!=ids[:len(records)] or len(records)>len(ids):raise RuntimeError('부분 예측 ID 중복/순서 불일치')
    return progress,records

def save_chunk(root,progress,rows):
    if not rows:return
    name=f'chunks/part_{len(progress["chunks"]):05d}.json';p=root/name;p.parent.mkdir(exist_ok=True)
    write(p,rows);progress['chunks'].append({'file':name,'sha256':sha(p)})
    write(root/'progress.json',progress)


def export(cfg):
    import pandas as pd
    root=Path(cfg['output_dir']);test,sample=tables(cfg);f=read(root/'frozen_inputs.json')
    if f['fingerprint']!=cfg['fingerprint'] or sha(cfg['test_csv'])!=f['test_sha256'] or sha(cfg['sample_csv'])!=f['sample_sha256']:raise RuntimeError('입력 CSV 변경')
    _,records=load_progress(root,cfg['fingerprint'],test.id.tolist())
    if len(records)!=len(test):raise RuntimeError(f'추론 미완료: {len(records)}/{len(test)}')
    preds=pd.DataFrame(records);preds.assign(image_meta=preds.image_meta.map(json.dumps)).to_csv(root/'test_predictions.csv',index=False,encoding='utf-8-sig')
    failures=preds[~preds.answer.isin(list('abcd'))];failures.to_csv(root/'parse_failures.csv',index=False,encoding='utf-8-sig')
    report={'n':len(test),'parse_failure_n':len(failures),'fallback_n':int(preds.fallback_used.sum()),'id_order_matches_sample':preds.id.tolist()==sample.id.tolist(),
            'checkpoint':f['checkpoint'],'auto_submitted':False,'review_02':'pending','test_accuracy':None}
    if len(failures) or report['fallback_n'] or not report['id_order_matches_sample']:
        report['state']='blocked';write(root/'submission_validation.json',report)
        raise RuntimeError('파싱/ID 검사 실패. parse_failures.csv 확인. 임의 답 채우기 없이 submission 생성 중단')
    result=sample.copy();result['answer']=preds.answer.tolist()
    tmp=root/'submission.csv.tmp';result.to_csv(tmp,index=False,encoding='utf-8')
    check=pd.read_csv(tmp,dtype=str,keep_default_na=False)
    if list(check.columns)!=list(sample.columns) or not check.equals(result):raise RuntimeError('CSV 재로드 검사 실패')
    final=root/'submission.csv'
    if final.exists() and sha(final)!=sha(tmp):raise RuntimeError('기존 submission 내용과 다릅니다. 기존 파일 보존 후 확인하세요.')
    tmp.replace(final);report.update(state='file_checks_passed',sha256=sha(final),path=str(final),answer_counts=check.answer.value_counts().to_dict())
    write(root/'submission_validation.json',report)
    (root/'PROJECT_STATUS_update.md').write_text('# TASK-006 전체 test 추론\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n\n제출 CSV 생성 및 파일 검사 완료. 독립 검토·실제 제출 미완료. 검증·holdout을 제외한 전체 학습 데이터로 새 학습. test 정답 미사용.',encoding='utf-8')
    print('submission 생성 완료:',final,flush=True);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)



def configure_pixels(adapter,pixels):
    ip=adapter.processor.image_processor;ip.size={'shortest_edge':pixels,'longest_edge':pixels}
    if hasattr(ip,'min_pixels'):ip.min_pixels=pixels
    if hasattr(ip,'max_pixels'):ip.max_pixels=pixels

def stage_record(cfg,name):
    return completed(Path(cfg['output_dir']),name)

def commit(cfg,name,out,result):
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(Path(cfg['output_dir'])/(name+'_complete.json'),{'directory':str(out),'artifacts':files,'result':result})

def split_data(cfg):
    return load_split(cfg,Path(cfg['output_dir']))

def prepare_colab(cfg):
    import pandas as pd
    from PIL import Image,ImageOps
    root=Path(cfg['output_dir']);test,sample=tables(cfg)
    if sha(cfg['test_csv'])!=cfg['test_sha256'] or sha(cfg['sample_csv'])!=cfg['sample_sha256']:raise RuntimeError('입력 CSV 변경')
    import shutil,pandas as pd
    base=Path(cfg['baseline_dir'])
    original=read(base/'run_config.json')
    _,original_valid,_=RUNTIME_BASE.load_split(original,base)
    old=stage_record(cfg,'audit')
    if old:split_data(cfg)
    else:
        original_audit=RUNTIME_BASE.completed(base,'audit');src=Path(original_audit['directory'])
        out=root/('audit_'+uuid.uuid4().hex[:8]);out.mkdir()
        for name in ['image_audit.csv','data_findings.csv','image_errors.csv','near_duplicate_candidates.csv']:
            if (src/name).is_file():shutil.copy2(src/name,out/name)
        split=pd.read_csv(src/'split_manifest.csv',dtype=str,keep_default_na=False)
        if not split.split.isin(['train','valid','holdout','pool']).all():raise ValueError('기존 split 이름 오류')
        preserved=split.loc[split.split.isin(['valid','holdout'])].copy()
        split.loc[split.split=='pool','split']='train'
        if not split.loc[split.split.isin(['valid','holdout'])].equals(preserved):raise RuntimeError('검증/holdout 변경')
        if split.groupby('group_id').split.nunique().max()!=1:raise RuntimeError('이미지 그룹 분할 누수')
        split.to_csv(out/'split_manifest.csv',index=False)
        for name in ['train','valid','holdout','pool']:split[split.split==name].to_csv(out/f'{name}_ids.csv',index=False)
        frame=pd.read_csv(Path(cfg['data_dir'])/'train.csv',dtype=str,keep_default_na=False).merge(split,on='id',validate='one_to_one')
        if len(frame)!=len(split) or set(frame.id)!=set(split.id):raise ValueError('train CSV와 분할 ID 불일치')
        pd.crosstab(frame.split,frame.answer).to_csv(out/'answer_distribution.csv');pd.crosstab(frame.split,frame.question_type).to_csv(out/'type_distribution.csv')
        info=read(src/'data_manifest.json')
        info.update(split_sha256=sha(out/'split_manifest.csv'),image_audit_sha256=sha(out/'image_audit.csv'),
                    counts=split.split.value_counts().to_dict(),split_source='original fixed valid/holdout; original train+pool for training')
        if info['counts'].get('valid')!=500:raise RuntimeError('원래 검증이 500개인지 확인하세요.')
        write(out/'data_manifest.json',info);commit(cfg,'audit',out,info)
    _,new_valid,_=split_data(cfg)
    if new_valid.id.tolist()!=original_valid.id.tolist():raise RuntimeError('검증 ID/순서 변경')
    info=stage_record(cfg,'audit')['result'];counts=info['counts']
    plan={'counts':counts,'epochs':1,'optimizer_updates':math.ceil(counts['train']/cfg['gradient_accumulation']),
          'reference_train_seconds_5060ti':2729.7907537*counts['train']/1000,
          'note':'linear estimate from local 1000-sample run; full training not yet measured'}
    write(root/'training_plan.json',plan);print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    images=[]
    for i,row in enumerate(test.to_dict('records')):
        p=resolve_image(cfg['data_dir'],row['path'])
        with Image.open(p) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB');rgb.load();size=list(rgb.size)
        images.append({'id':row['id'],'path':row['path'],'sha256':sha(p),'size':size})
        if (i+1)%500==0:print('test 이미지 확인',i+1,'/',len(test),flush=True)
    frozen={'fingerprint':cfg['fingerprint'],'test_sha256':sha(cfg['test_csv']),'sample_sha256':sha(cfg['sample_csv']),
            'images':images,'n':len(test),'revision':cfg['revision'],'checkpoint':'pending'}
    path=root/'frozen_inputs.json'
    if path.exists():
        prior=read(path);frozen['checkpoint']=prior['checkpoint']
        if frozen!=prior:raise RuntimeError('입력 변경. SESSION_TAG 변경 필요')
    write(path,frozen)
    write(root/'prompt.json',{'system':SYSTEM_INSTRUCT,'instruction':cfg['p5_instruction'],'name':'P5'})
    print('입력 준비 완료:',len(test),'test 문항',flush=True)

def train_colab(cfg):
    import torch
    root=Path(cfg['output_dir'])
    if stage_record(cfg,'train'):print('완료된 학습/모델 선택 재사용',flush=True);return
    tr,va,info=split_data(cfg);out=root/('train_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir()
    gpu_environment(cfg,out)
    install_answer_collator(RUNTIME_BASE)
    adapter=ModelAdapter(cfg,out);adapter.add_lora();write(out/'training_config.json',cfg)
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
    audit_targets(sys.modules[__name__],adapter,tr,out)
    # Small forward/backward check before optimizer training; reset RNG afterwards.
    row=tr.iloc[0].to_dict();batch=adapter.encode(row,training=True);adapter.model.train()
    with torch.autocast('cuda',dtype=adapter.dtype):loss=adapter.loss(batch)
    if not torch.isfinite(loss):raise FloatingPointError('smoke loss NaN/Inf')
    loss.backward();grads=[p.grad for p in adapter.model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads) or not any(bool(g.abs().max()>0) for g in grads):raise FloatingPointError('smoke gradient 오류')
    write(out/'smoke.json',{'loss':float(loss.detach()),'finite_nonzero_gradients':True,'optimizer_step':False})
    adapter.model.zero_grad(set_to_none=True);del loss,batch,grads;gc.collect();torch.cuda.empty_cache()
    # Baseline initialization consumes RNG before dropout. Preserve the post-initialization state below.
    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
    result=train_one_epoch(adapter,tr,cfg,out)
    checkpoint=out/'adapter_epoch1';adapter.model.save_pretrained(checkpoint);adapter.processor.save_pretrained(checkpoint)
    configure_pixels(adapter,672**2)
    before,_=evaluate(adapter,va.head(5),'reload_before',out);adapter.reload_adapter(checkpoint)
    after,_=evaluate(adapter,va.head(5),'reload_after',out)
    same=len(before)==len(after) and all(all(x[k]==y[k] for k in ['id','answer','raw_output']) for x,y in zip(before,after))
    write(out/'reload_check.json',{'identical_outputs':same,'n':len(before)})
    if not same:raise RuntimeError('저장/재로드 출력 불일치')
    result['checkpoint']=str(checkpoint);commit(cfg,'train',out,result)

def validate_colab(cfg):
    if cfg['mode']=='load_adapter':print('업로드 모델은 새 검증 분할로 독립 Accuracy를 주장하지 않습니다.',flush=True);return
    if stage_record(cfg,'validation'):print('완료 검증 재사용',flush=True);return
    from peft import PeftModel
    tr,va,info=split_data(cfg);root=Path(cfg['output_dir']);rec=stage_record(cfg,'train')
    if rec is None:raise RuntimeError('학습 먼저 실행')
    out=root/('validation_'+uuid.uuid4().hex[:8]);out.mkdir();settings=dict(cfg);settings['pixel_budget']=672**2
    gpu_environment(settings,out);adapter=ModelAdapter(settings,out)
    adapter.model=PeftModel.from_pretrained(adapter.model,rec['result']['checkpoint'],local_files_only=True,is_trainable=False)
    adapter.generate(adapter.encode(va.iloc[0].to_dict(),training=False))
    _,metrics=evaluate(adapter,va,'valid',out);commit(cfg,'validation',out,metrics)
    print('전체 학습 검증 Accuracy:',metrics['accuracy'],flush=True)

def infer_colab(cfg):
    import torch
    from peft import PeftModel
    root=Path(cfg['output_dir']);f=read(root/'frozen_inputs.json');test,sample=tables(cfg)
    if f['fingerprint']!=cfg['fingerprint'] or sha(cfg['test_csv'])!=f['test_sha256'] or sha(cfg['sample_csv'])!=f['sample_sha256']:raise RuntimeError('CSV 변경')
    for r in f['images']:
        if sha(resolve_image(cfg['data_dir'],r['path']))!=r['sha256']:raise RuntimeError('test 이미지 변경')
    rec=stage_record(cfg,'train')
    if rec is None:raise RuntimeError('학습/모델 선택 먼저 실행')
    checkpoint=rec['result']['checkpoint'];f['checkpoint']=checkpoint;write(root/'frozen_inputs.json',f)
    progress,records=load_progress(root,cfg['fingerprint'],test.id.tolist())
    if len(records)==len(test):print('전체 추론 완료 기록 재사용',flush=True);return
    out=root/('infer_'+uuid.uuid4().hex[:8]);out.mkdir();settings=dict(cfg);settings['pixel_budget']=672**2
    gpu_environment(settings,out);adapter=ModelAdapter(settings,out)
    adapter.model=PeftModel.from_pretrained(adapter.model,checkpoint,local_files_only=True,is_trainable=False)
    ac=read(Path(checkpoint)/'adapter_config.json')
    if ac['r']!=8 or ac['lora_alpha']!=16 or ac['lora_dropout']!=.05:raise RuntimeError('LoRA 설정 불일치')
    adapter.generate(adapter.encode(test.iloc[len(records)].to_dict(),training=False));torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();pending=[];before=len(records)
    try:
        for row in test.iloc[before:].to_dict('records'):
            tick=time.perf_counter();enc=adapter.encode(row,training=False);raw=adapter.generate(enc);torch.cuda.synchronize()
            ans=parse_answer(raw) or ''
            pending.append({'id':row['id'],'answer':ans,'raw_output':raw,'parse_failed':not bool(ans),'fallback_used':False,
                            'seconds':time.perf_counter()-tick,'image_meta':adapter.last_image_meta});del enc
            if len(pending)>=cfg['save_every']:
                save_chunk(root,progress,pending);records.extend(pending);pending=[]
                rate=(time.perf_counter()-start)/(len(records)-before)
                print(f'test {len(records)}/{len(test)} | {rate:.3f}초/문항 | 예상 잔여 {(len(test)-len(records))*rate/60:.1f}분',flush=True)
    finally:
        save_chunk(root,progress,pending)
        write(out/'runtime.json',{'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})


import contextlib,importlib.util
def alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
        k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];k.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=k.OpenProcess(0x1000,False,pid)
        if not handle:return False if ctypes.get_last_error()==87 else None
        try:
            code=wintypes.DWORD()
            if not k.GetExitCodeProcess(handle,ctypes.byref(code)):return None
            return code.value==259
        finally:k.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return None

@contextlib.contextmanager
def lock_gpu(cfg):
    lock=Path(cfg['project_dir'])/'output/TASK-006-local-alltrain/gpu_experiment.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    for name in ['TASK-006-lr','TASK-006-loss','TASK-006-prompts','TASK-006-resolution','TASK-006-training-resolution']:
        other=Path(cfg['project_dir'])/'output'/name/'gpu_experiment.lock'
        if other.exists():raise RuntimeError(f'기존 실험 잠금: {other}. 다른 학습/추론이 없는지 확인하세요.')
    lr=read(Path(cfg['lr_dir'])/'resolution_config.json')
    if (Path(lr['baseline_dir'])/'running.lock').exists():raise RuntimeError('baseline 실행 잠금이 남아 있습니다.')
    if lock.exists():
        old=read(lock)
        if alive(int(old['pid'])) is False:lock.unlink()
        else:raise RuntimeError(f'제출 추론 프로세스 실행 중이거나 확인 불가: PID {old["pid"]}')
    token=uuid.uuid4().hex
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'token':token},f)
    try:yield
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()



def load_runtime(cfg):
    global RUNTIME_BASE,ModelAdapter,train_one_epoch,evaluate,gpu_environment,parse_answer
    for path,digest in cfg['source_hashes'].items():
        if sha(path)!=digest:raise RuntimeError(f'기준 파일 변경: {path}')
    base=Path(cfg['baseline_dir']);bc=read(base/'run_config.json')
    worker=base/'task006_worker.py';data=worker.read_bytes()
    if bc['worker_sha256'] not in [hashlib.sha256(data).hexdigest(),hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()]:raise RuntimeError('baseline 코드 내용 변경')
    spec=importlib.util.spec_from_file_location('actual_baseline_worker',worker);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.block_network()
    def p5(question,a,b,c,d):return f'{question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n'+cfg['p5_instruction']
    m.build_mc_prompt=p5
    RUNTIME_BASE=m;ModelAdapter=m.ModelAdapter;train_one_epoch=m.train_one_epoch;evaluate=m.evaluate;gpu_environment=m.gpu_environment;parse_answer=m.parse_answer
    if cfg['mode']!='train_fresh' or cfg['train_n']!='all_eligible':raise ValueError('전체 새 학습 설정 필요')
    if cfg['pixel_budget']!=640**2 or cfg['learning_rate']!=1e-4 or cfg['loss_mode']!='answer_only' or cfg['prompt']!='P5':raise RuntimeError('고정 조건 불일치')
    if cfg['test_sha256']!=sha(cfg['test_csv']) or cfg['sample_sha256']!=sha(cfg['sample_csv']):raise RuntimeError('test/sample 변경')
    if cfg['split_sha256']!=sha(cfg['split_manifest']):raise RuntimeError('기존 분할 변경')
    rec=read(Path(cfg['lr_dir'])/'res_lr_1e-4_status.json')
    if rec['state']!='completed':raise RuntimeError('기준 LR 평가 미완료')
    for name,h in rec['artifacts'].items():
        if sha(Path(rec['directory'])/name)!=h:raise RuntimeError('기준 평가 결과 변경')

def comparison(cfg):
    import pandas as pd
    root=Path(cfg['output_dir']);new=stage_record(cfg,'validation')
    if not new:return
    old=read(Path(cfg['lr_dir'])/'res_lr_1e-4_status.json');old_dir=Path(old['directory']);new_dir=Path(new['directory'])
    a=pd.read_csv(old_dir/'valid_predictions.csv',dtype=str,keep_default_na=False)
    b=pd.read_csv(new_dir/'valid_predictions.csv',dtype=str,keep_default_na=False)
    m=a.merge(b,on='id',validate='one_to_one',suffixes=('_1000','_all'))
    if len(m)!=len(a) or len(m)!=len(b) or not (m.gold_1000==m.gold_all).all() or not (m.group_id_1000==m.group_id_all).all():raise RuntimeError('비교 문항 불일치')
    x=m.answer_1000==m.gold_1000;y=m.answer_all==m.gold_all
    m['transition']=['gain' if v and not u else 'loss' if u and not v else 'same_correct' if u else 'same_wrong' for u,v in zip(x,y)]
    m.to_csv(root/'paired_1000_vs_all.csv',index=False,encoding='utf-8-sig')
    m[m.transition.isin(['gain','loss'])].to_csv(root/'changed_1000_vs_all.csv',index=False,encoding='utf-8-sig')
    metrics=read(old_dir/'valid_metrics.json');training=stage_record(cfg,'train')['result']
    rows=[{'condition':'previous_1000','train_n':cfg['reference_train_n'],**metrics},
          {'condition':'all_eligible','train_n':training['train_n'],**new['result'],'training_seconds':training['seconds'],
           'optimizer_updates':training['updates'],'gain':int((~x&y).sum()),'loss':int((x&~y).sum()),'delta_pp':100*float(y.mean()-x.mean())}]
    for row in rows:row['accuracy_pct']=100*row['accuracy']
    table=pd.DataFrame(rows);table.to_csv(root/'data_size_comparison.csv',index=False,encoding='utf-8-sig')
    print(table[['condition','train_n','correct_n','accuracy_pct','gain','loss','delta_pp']].to_string(index=False),flush=True)

def run_local(cfg,stage):
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=True)
    with lock_gpu(cfg):
        load_runtime(cfg)
        try:
            if stage=='prepare':
                assets=read(Path(cfg['baseline_dir'])/'model_assets.json')
                if assets['revision']!=cfg['revision']:raise RuntimeError('revision 변경')
                for name,item in assets['files'].items():
                    if sha(Path(cfg['model_dir'])/name)!=item['sha256']:raise RuntimeError(f'모델 파일 변경: {name}')
            {'prepare':prepare_colab,'train':train_colab,'validation':validate_colab,'infer':infer_colab,'export':export}[stage](cfg)
            if stage=='validation':comparison(cfg)
            with open(root/'CHANGELOG.md','a',encoding='utf-8') as f:f.write(f'\n- {time.strftime("%Y-%m-%d %H:%M:%S")} {stage}: completed\n')
        except BaseException as exc:write(root/(stage+'_last_error.json'),{'type':type(exc).__name__,'message':str(exc)});raise

if __name__=='__main__':run_local(read(sys.argv[1]),sys.argv[2])
