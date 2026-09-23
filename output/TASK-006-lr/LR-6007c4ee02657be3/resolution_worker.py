import csv, hashlib, importlib.util, json, os, sys, time, uuid
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_worker_source(path, expected):
    data=Path(path).read_bytes()
    raw=hashlib.sha256(data).hexdigest()
    lf=hashlib.sha256(data.replace(b"\r\n",b"\n")).hexdigest()
    if expected not in (raw,lf):
        raise RuntimeError(f"baseline 코드 내용이 다릅니다. expected={expected}, raw={raw}, LF={lf}. 기존 파일/설정을 수정하지 말고 실행 폴더를 확인하세요.")
    return expected


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,obj):
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8');tmp.replace(p)


def verified_status(root,size):
    p=root/f'res_{size}_status.json'
    if not p.exists(): return {'state':'not_run'}
    rec=read(p)
    if rec['state']=='completed':
        for name,sha in rec['artifacts'].items():
            f=Path(rec['directory'])/name
            if not f.is_file() or digest(f)!=sha: raise RuntimeError(f'완료 결과 변경/누락: {f}')
    return rec


def paired(a,b):
    m=a.merge(b,on='id',validate='one_to_one',suffixes=('_reference','_new'))
    if len(m)!=len(a) or len(m)!=len(b) or not (m.gold_reference==m.gold_new).all():
        raise RuntimeError('비교 ID/정답이 동일하지 않습니다.')
    if not (m.group_id_reference==m.group_id_new).all(): raise RuntimeError('이미지 그룹 불일치')
    old=m.answer_reference==m.gold_reference;new=m.answer_new==m.gold_new
    m['transition']=['gain' if y and not x else 'loss' if x and not y else 'same_correct' if x else 'same_wrong' for x,y in zip(old,new)]
    return m,{'gain':int((~old & new).sum()),'loss':int((old & ~new).sum()),
              'delta_pp':100*float(new.mean()-old.mean())}


def summarize(cfg):
    import pandas as pd
    root=Path(cfg['output_dir']);rows=[];preds={};types=[]
    for name in cfg['resolutions']:
        rec=verified_status(root,name)
        r={'condition':name,'learning_rate':cfg['learning_rates'][name],'status':rec['state'],
           'loss_mode':'answer_only','prompt':'P5','train_resolution':640,'inference_resolution':672}
        tr=root/f'train_{name}_status.json'
        if tr.exists():
            t=read(tr);r['train_status']=t['state']
            if t['state']=='completed':r['training_seconds']=t['training']['seconds']
        if rec['state']=='completed':
            d=Path(rec['directory']);r.update(read(d/'valid_metrics.json'));r['accuracy_pct']=100*r['accuracy']
            r['parse_failure_pct']=100*r['parse_failure_rate']
            preds[name]=pd.read_csv(d/'valid_predictions.csv',keep_default_na=False)
        else:r['error']=rec.get('error','')
        rows.append(r)
    for r in rows:
        name=r['condition']
        if name not in preds:continue
        if 'lr_1e-4' in preds:
            m,stats=paired(preds['lr_1e-4'],preds[name]);r.update(stats)
            m.to_csv(root/f'paired_1e-4_vs_{name}.csv',index=False,encoding='utf-8-sig')
            m[m.transition.isin(['gain','loss'])].to_csv(root/f'changed_1e-4_vs_{name}.csv',index=False,encoding='utf-8-sig')
        for kind,g in preds[name].groupby('question_type'):
            types.append({'condition':name,'question_type':kind,'n':len(g),'accuracy':float((g.answer==g.gold).mean())})
    pd.DataFrame(types).to_csv(root/'type_comparison.csv',index=False,encoding='utf-8-sig')
    table=pd.DataFrame(rows);table.to_csv(root/'lr_comparison.csv',index=False,encoding='utf-8-sig')
    write(root/'summary.json',{'results':rows,'adoption':'pending','review_02':'pending','test_used':False,'holdout_evaluated':False})
    (root/'PROJECT_STATUS_update.md').write_text('# TASK-006 LR / P5\n\n'+table.to_string(index=False)+'\n\n채택·독립 검토 미완료. P5 학습을 새로 수행한 LR 비교.',encoding='utf-8')
    print(table.to_string(index=False),flush=True)
    return rows


def load_baseline(cfg):
    base=Path(cfg['baseline_dir']);original=read(base/'run_config.json')
    worker=base/'task006_worker.py'
    if verify_worker_source(worker,original['worker_sha256'])!=cfg['baseline_worker_sha256']:
        raise RuntimeError('baseline 실행 코드 해시 불일치')
    spec=importlib.util.spec_from_file_location('saved_baseline',worker)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.block_network()
    def p5_prompt(question,a,b,c,d):
        return f'{question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n'+cfg['p5_instruction']
    module.build_mc_prompt=p5_prompt
    return module,original


def prepare(cfg):
    import pandas as pd
    root=Path(cfg['output_dir']);base=Path(cfg['baseline_dir']);module,original=load_baseline(cfg)
    train=module.completed(base,'train');evaluation=module.completed(base,'lora_eval')
    audit=module.completed(base,'audit')
    if train is None or evaluation is None or audit is None:raise RuntimeError('baseline audit/train/lora_eval을 먼저 완료하세요.')
    if original['pixel_budget']!=384**2:raise RuntimeError('학습 해상도 384²인 baseline을 지정하세요.')
    reload_check=read(Path(train['directory'])/'reload_check.json')
    if not reload_check.get('identical_outputs'):raise RuntimeError('baseline 저장/재로드 검증 미통과')
    fixed_train,valid,info=module.load_split(original,base)
    if len(valid)!=500:raise RuntimeError(f'고정 검증 500개가 아닙니다: {len(valid)}. 기존 분할 확인 필요')
    if original['epochs']!=1:raise RuntimeError('이번 코드는 1 epoch 비교입니다.')
    loss_dir=Path(cfg['loss_dir']);record_path=loss_dir/'train_answer_only_status.json'
    if digest(record_path)!=cfg['loss_record_sha256']:raise RuntimeError('선택 loss 기록 변경')
    rec=read(record_path)
    if rec['state']!='completed':raise RuntimeError('answer_only 학습 미완료')
    for name,h in rec['artifacts'].items():
        if digest(Path(rec['directory'])/name)!=h:raise RuntimeError('answer_only 학습 artifact 변경')
    prior=read(Path(rec['directory'])/'training_config.json')
    for key in original:
        if key not in ['run_dir','pixel_budget','image_policy','loss','loss_mode'] and prior.get(key)!=original[key]:
            raise RuntimeError(f'선택 학습과 baseline 설정 차이: {key}')
    if prior['pixel_budget']!=640**2 or prior['loss_mode']!='answer_only':raise RuntimeError('선택 loss 조건 불일치')
    lf=read(loss_dir/'frozen_inputs.json')
    if lf['data_manifest']!=info or lf['baseline_config_sha256']!=digest(base/'run_config.json'):raise RuntimeError('선택 loss의 데이터/기준 설정 불일치')
    if not read(Path(rec['directory'])/'reload_check.json').get('identical_outputs'):raise RuntimeError('선택 loss 재로드 미검증')
    fixed_train[['id','group_id','question_type']].to_csv(root/'fixed_train_ids.csv',index=False)
    write(root/'prompt.json',{'name':'P5','instruction':cfg['p5_instruction'],'system':module.SYSTEM_INSTRUCT,
         'example':module.build_mc_prompt(*(fixed_train.iloc[0][k] for k in ['question','a','b','c','d']))})
    print('고정 학습:',len(fixed_train),'검증:',len(valid),'새 학습 3회, P5 train + inference',flush=True)
    # Avoid relying only on paths; check every model asset against baseline download hashes.
    assets=read(base/'model_assets.json')
    if assets['revision']!=original['revision']:raise RuntimeError('모델 revision 불일치')
    for name,record in assets['files'].items():
        f=Path(original['model_dir'])/name
        if not f.is_file() or digest(f)!=record['sha256']:raise RuntimeError(f'모델 파일 확인 필요: {name}')
    valid[['id','group_id','question_type']].to_csv(root/'fixed_valid_ids.csv',index=False)
    frozen={'baseline_config':original,'data_manifest':info,'checkpoint':str(Path(train['directory'])/'adapter_epoch1'),
            'baseline_records':{name:digest(base/(name+'_complete.json')) for name in ['audit','train','lora_eval']},
            'baseline_config_sha256':digest(base/'run_config.json'),'valid_ids_sha256':digest(root/'fixed_valid_ids.csv'),
            'train_ids_sha256':digest(root/'fixed_train_ids.csv'),'train_n':len(fixed_train),'valid_n':len(valid),'previous_predictions':str(Path(evaluation['directory'])/'valid_lora_predictions.csv'),
            'model_file_stats':{name:[(Path(original['model_dir'])/name).stat().st_size,(Path(original['model_dir'])/name).stat().st_mtime_ns] for name in assets['files']}}
    frozen_path=root/'frozen_inputs.json'
    if frozen_path.exists() and read(frozen_path)!=frozen:raise RuntimeError('기준 입력 변경. 새 SESSION_TAG로 실행하세요.')
    write(frozen_path,frozen)
    print('준비 완료. 이력 LoRA (학습에 재사용하지 않음):',frozen['checkpoint'],'검증:',len(valid),flush=True)
    summarize(cfg)


def configure_pixels(adapter,pixels):
    ip=adapter.processor.image_processor
    ip.size={'shortest_edge':pixels,'longest_edge':pixels}
    if hasattr(ip,'min_pixels'):ip.min_pixels=pixels
    if hasattr(ip,'max_pixels'):ip.max_pixels=pixels


def validate_inputs(cfg):
    import pandas as pd
    root=Path(cfg['output_dir']);base=Path(cfg['baseline_dir']);frozen=read(root/'frozen_inputs.json')
    module,original=load_baseline(cfg)
    if digest(base/'run_config.json')!=frozen['baseline_config_sha256']:raise RuntimeError('baseline 설정 변경')
    for name,sha in frozen['baseline_records'].items():
        if digest(base/(name+'_complete.json'))!=sha:raise RuntimeError('baseline 완료 기록 변경')
    module.completed(base,'train')
    train,valid,info=module.load_split(original,base)
    if digest(root/'fixed_train_ids.csv')!=frozen['train_ids_sha256'] or pd.read_csv(root/'fixed_train_ids.csv',dtype=str).id.tolist()!=train.id.tolist():raise RuntimeError('학습 ID/순서 변경')
    if info!=frozen['data_manifest']:raise RuntimeError('분할 manifest 변경')
    if digest(root/'fixed_valid_ids.csv')!=frozen['valid_ids_sha256']:raise RuntimeError('검증 ID 파일 변경')
    if pd.read_csv(root/'fixed_valid_ids.csv',dtype=str).id.tolist()!=valid.id.tolist():raise RuntimeError('검증 순서 변경')
    for name,stat in frozen['model_file_stats'].items():
        f=Path(original['model_dir'])/name
        if [f.stat().st_size,f.stat().st_mtime_ns]!=stat:raise RuntimeError('모델 파일 변경')
    return module,original,frozen,train,valid


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


def train_condition(cfg,size):
    import torch
    root=Path(cfg['output_dir']);module,original,frozen,train,valid=validate_inputs(cfg)
    status=root/f'train_{size}_status.json'
    if status.exists():
        old=read(status)
        if old['state']=='completed':
            for name,h in old.get('artifacts',{}).items():
                if digest(Path(old['directory'])/name)!=h:raise RuntimeError('학습 결과 파일 변경')
            print('완료 학습 재사용:',size,flush=True);return
        if old['state']=='blocked_oom' and not cfg['retry_oom']:return
    out=root/f'train_{size}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}';out.mkdir()
    settings=dict(original);settings['pixel_budget']=640**2;settings['run_dir']=str(out)
    settings['image_policy']='train pixel budget 640 squared'
    settings['loss_mode']='answer_only';settings['loss']='answer_and_eos_only'
    settings['learning_rate']=cfg['learning_rates'][size];settings['prompt']='P5';settings['p5_instruction']=cfg['p5_instruction']
    write(out/'training_config.json',settings);write(status,{'state':'running','directory':str(out)})
    try:
        module.gpu_environment(settings,out)
        install_answer_collator(module)
        adapter=module.ModelAdapter(settings,out)
        adapter.add_lora()  # Fresh base and fresh LoRA; no preceding resolution adapter loaded.
        audit_targets(module,adapter,train,out)
        training=module.train_one_epoch(adapter,train,settings,out)
        checkpoint=out/'adapter_epoch1'
        adapter.model.save_pretrained(checkpoint);adapter.processor.save_pretrained(checkpoint)
        # Use fixed inference budget for both sides of serialization check.
        configure_pixels(adapter,cfg['inference_resolution']**2)
        probe=valid.head(min(5,len(valid)))
        before,_=module.evaluate(adapter,probe,'reload_before',out)
        adapter.reload_adapter(checkpoint)
        after,_=module.evaluate(adapter,probe,'reload_after',out)
        keys=['id','answer','raw_output']
        identical=all(all(x[k]==y[k] for k in keys) for x,y in zip(before,after)) and len(before)==len(after)
        write(out/'reload_check.json',{'n':len(probe),'identical_outputs':identical})
        if not identical:raise RuntimeError('저장/재로드 예측 불일치. 결과 검토 필요')
        artifacts={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
        write(status,{'state':'completed','directory':str(out),'checkpoint':str(checkpoint),
                      'training':training,'artifacts':artifacts,'reused_baseline':False})
    except torch.cuda.OutOfMemoryError as exc:
        write(status,{'state':'blocked_oom','directory':str(out),'error':str(exc)})
    except BaseException as exc:
        write(status,{'state':'failed_or_interrupted','directory':str(out),'error':str(exc)});raise


def run_resolution(cfg,size):
    import torch
    from peft import PeftModel
    root=Path(cfg['output_dir']);module,original,frozen,train,valid=validate_inputs(cfg)
    old=verified_status(root,size)
    if old['state']=='completed' or (old['state']=='blocked_oom' and not cfg['retry_oom']):
        summarize(cfg);return
    trained=read(root/f'train_{size}_status.json')
    status=root/f'res_{size}_status.json'
    if trained['state']!='completed':
        write(status,{'state':trained['state'],'error':trained.get('error','training incomplete')});summarize(cfg);return
    for name,h in trained.get('artifacts',{}).items():
        if digest(Path(trained['directory'])/name)!=h:raise RuntimeError('학습 artifact 변경')
    out=root/f'eval_train{size}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}';out.mkdir()
    settings=dict(original);settings['pixel_budget']=cfg['inference_resolution']**2;settings['run_dir']=str(out)
    settings.update(loss_mode='answer_only',learning_rate=cfg['learning_rates'][size],prompt='P5',p5_instruction=cfg['p5_instruction'])
    write(out/'inference_config.json',settings);write(status,{'state':'running','directory':str(out)})
    try:
        module.gpu_environment(settings,out);adapter=module.ModelAdapter(settings,out)
        adapter.model=PeftModel.from_pretrained(adapter.model,trained['checkpoint'],local_files_only=True,is_trainable=False)
        probe=valid.iloc[0].to_dict();probe.pop('answer',None)
        adapter.generate(adapter.encode(probe,training=False));torch.cuda.synchronize()
        _,metrics=module.evaluate(adapter,valid,'valid',out)
        metrics.update({'train_resolution':640,'loss_mode':'answer_only','learning_rate':cfg['learning_rates'][size],'prompt':'P5','inference_resolution':cfg['inference_resolution'],
            'checkpoint':trained['checkpoint'],'training_seconds':trained['training']['seconds'],
            'optimizer_updates':trained['training']['updates'],
            'training_peak_allocated_gib':trained['training']['peak_allocated_gib'],
            'reused_baseline':trained['reused_baseline'],'warmup_samples':1})
        write(out/'valid_metrics.json',metrics)
        artifacts={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
        write(status,{'state':'completed','directory':str(out),'artifacts':artifacts})
    except torch.cuda.OutOfMemoryError as exc:
        write(status,{'state':'blocked_oom','directory':str(out),'error':str(exc)})
    except BaseException as exc:
        write(status,{'state':'failed_or_interrupted','directory':str(out),'error':str(exc)});raise
    finally:
        with open(root/'CHANGELOG.md','a',encoding='utf-8') as f:f.write(f"\n- train {size}, inference {cfg['inference_resolution']}: {read(status)['state']}\n")
    summarize(cfg)


if __name__=='__main__':
    config=read(sys.argv[1]);stage=sys.argv[2]
    if stage=='prepare':prepare(config)
    elif stage=='summary':summarize(config)
    else:
        is_train=stage.startswith('train_')
        size=stage.removeprefix('train_')
        if size not in config['resolutions']:raise ValueError('설정에 없는 해상도')
        if is_train:train_condition(config,size)
        else:run_resolution(config,size)
