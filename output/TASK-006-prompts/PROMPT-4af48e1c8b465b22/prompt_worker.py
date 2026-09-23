
import hashlib,importlib.util,json,os,sys,time,uuid
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8');t.replace(p)
def check_artifacts(record):
    if record['state']!='completed':raise RuntimeError('선택한 학습이 완료되지 않았습니다.')
    for name,h in record.get('artifacts',{}).items():
        p=Path(record['directory'])/name
        if not p.is_file() or sha(p)!=h:raise RuntimeError(f'완료된 학습 결과 변경/누락: {p}')

def context(cfg):
    base=Path(cfg['baseline_dir']);bc=read(base/'run_config.json');path=base/'task006_worker.py'
    data=path.read_bytes();raw=hashlib.sha256(data).hexdigest();lf=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()
    if bc['worker_sha256'] not in (raw,lf) or bc['worker_sha256']!=cfg['baseline_worker_sha256']:raise RuntimeError('baseline 코드 변경')
    if sha(base/'run_config.json')!=cfg['baseline_config_sha256']:raise RuntimeError('baseline 설정 변경')
    spec=importlib.util.spec_from_file_location('saved_baseline',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.block_network()
    record_path=Path(cfg['loss_dir'])/f"train_{cfg['loss_mode']}_status.json"
    if sha(record_path)!=cfg['train_record_sha256']:raise RuntimeError('선택 학습 기록 변경')
    record=read(record_path);check_artifacts(record)
    settings=read(Path(record['directory'])/'training_config.json')
    for key in ['seed','learning_rate','epochs','gradient_accumulation','model_id','revision']:
        if settings[key]!=bc[key]:raise RuntimeError(f'기준 학습과 불일치: {key}')
    if settings['pixel_budget']!=640**2 or settings['loss_mode']!=cfg['loss_mode']:raise RuntimeError('학습640/선택 loss를 확인하세요.')
    reload=read(Path(record['directory'])/'reload_check.json')
    if not reload.get('identical_outputs'):raise RuntimeError('저장/재로드 검증 미통과')
    loss_frozen=read(Path(cfg['loss_dir'])/'frozen_inputs.json')
    if loss_frozen['baseline_config_sha256']!=cfg['baseline_config_sha256']:raise RuntimeError('다른 baseline에서 생성된 loss 실험입니다.')
    _,valid,info=m.load_split(bc,base)
    if loss_frozen['data_manifest']!=info:raise RuntimeError('학습/검증 분할 불일치')
    return m,bc,record,valid,info

def status(cfg,name):
    p=Path(cfg['output_dir'])/f'{name}_status.json'
    if not p.exists():return {'state':'not_run'}
    rec=read(p)
    if rec['state']=='completed':check_artifacts(rec)
    return rec

def pair(a,b):
    x=a.merge(b,on='id',suffixes=('_P0','_new'),validate='one_to_one')
    if len(x)!=len(a) or len(x)!=len(b) or not (x.gold_P0==x.gold_new).all() or not (x.group_id_P0==x.group_id_new).all():raise RuntimeError('비교 대상 ID/정답/그룹 불일치')
    old=x.answer_P0==x.gold_P0;new=x.answer_new==x.gold_new
    x['transition']=['gain' if v and not u else 'loss' if u and not v else 'same_correct' if u else 'same_wrong' for u,v in zip(old,new)]
    return x,{'gain':int((~old&new).sum()),'loss':int((old&~new).sum()),'delta_pp':100*float(new.mean()-old.mean())}

def summary(cfg):
    import pandas as pd
    root=Path(cfg['output_dir']);rows=[];types=[];reference=None
    s=status(cfg,'P0')
    if s['state']=='completed':reference=pd.read_csv(Path(s['directory'])/'valid_predictions.csv',keep_default_na=False)
    for name in cfg['prompt_order']:
        s=status(cfg,name);r={'prompt':name,'title':cfg['prompts'][name]['title'],'status':s['state']}
        if s['state']=='completed':
            out=Path(s['directory']);r.update(read(out/'valid_metrics.json'));r['accuracy_pct']=100*r['accuracy'];r['parse_failure_pct']=100*r['parse_failure_rate']
            p=pd.read_csv(out/'valid_predictions.csv',keep_default_na=False)
            if reference is not None:
                x,stats=pair(reference,p);r.update(stats)
                x.to_csv(root/f'paired_P0_vs_{name}.csv',index=False,encoding='utf-8-sig')
                x[x.transition.isin(['gain','loss'])].to_csv(root/f'changed_P0_vs_{name}.csv',index=False,encoding='utf-8-sig')
            for kind,g in p.groupby('question_type'):
                types.append({'prompt':name,'question_type':kind,'n':len(g),'accuracy':float((g.answer==g.gold).mean())})
        else:r['error']=s.get('error','')
        rows.append(r)
    table=pd.DataFrame(rows);table.to_csv(root/'prompt_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(types,columns=['prompt','question_type','n','accuracy']).to_csv(root/'type_comparison.csv',index=False,encoding='utf-8-sig')
    write(root/'summary.json',{'results':rows,'loss_mode':cfg['loss_mode'],'adoption':'pending','test_used':False,'holdout_evaluated':False})
    (root/'PROJECT_STATUS_update.md').write_text('# 추론 프롬프트 비교\n\n동일 학습640 모델/추론672/고정 검증. 재학습 없음.\n\n'+table.to_string(index=False)+'\n\n채택/독립 검토 미완료.',encoding='utf-8')
    print(table.to_string(index=False),flush=True)
    return rows

def prepare(cfg):
    m,bc,record,valid,info=context(cfg);root=Path(cfg['output_dir'])
    assets=read(Path(cfg['baseline_dir'])/'model_assets.json')
    if assets['revision']!=bc['revision']:raise RuntimeError('revision 불일치')
    for name,meta in assets['files'].items():
        if sha(Path(bc['model_dir'])/name)!=meta['sha256']:raise RuntimeError('모델 파일 변경: '+name)
    valid[['id','group_id','question_type']].to_csv(root/'fixed_valid_ids.csv',index=False)
    write(root/'prepared.json',{'checkpoint':record['checkpoint'],'valid_n':len(valid),'data_manifest':info,
        'config_sha256':sha(root/'prompt_config.json'),'valid_ids_sha256':sha(root/'fixed_valid_ids.csv'),
        'model_stats':{n:[(Path(bc['model_dir'])/n).stat().st_size,(Path(bc['model_dir'])/n).stat().st_mtime_ns] for n in assets['files']}})
    write(root/'prompts.json',cfg['prompts']);print('준비 완료:',record['checkpoint'],'검증',len(valid),flush=True);summary(cfg)

def run(cfg,name):
    import torch
    from peft import PeftModel
    root=Path(cfg['output_dir']);frozen=read(root/'prepared.json')
    if sha(root/'prompt_config.json')!=frozen['config_sha256']:raise RuntimeError('준비 후 설정 변경')
    m,bc,record,valid,info=context(cfg)
    if sha(root/'fixed_valid_ids.csv')!=frozen['valid_ids_sha256'] or info!=frozen['data_manifest']:raise RuntimeError('검증 데이터 변경')
    for n,stats in frozen['model_stats'].items():
        p=Path(bc['model_dir'])/n
        if [p.stat().st_size,p.stat().st_mtime_ns]!=stats:raise RuntimeError('모델 파일 변경: 준비 단계 재실행 필요')
    previous=status(cfg,name)
    if previous['state'] in ['completed','blocked_oom']:
        print('기록 재사용:',name,previous['state'],flush=True);summary(cfg);return
    out=root/f'{name}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}';out.mkdir()
    state_path=root/f'{name}_status.json';write(state_path,{'state':'running','directory':str(out)})
    settings=dict(bc);settings['pixel_budget']=672**2;settings['run_dir']=str(out)
    prompt=cfg['prompts'][name]
    original=m.build_mc_prompt
    def custom(question,a,b,c,d):
        return (f'{question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n'+prompt['instruction'])
    if name!='P0':m.build_mc_prompt=custom
    sample=valid.iloc[0]
    rendered=m.build_mc_prompt(*(sample[k] for k in ['question','a','b','c','d']))
    if name=='P0' and rendered!=original(*(sample[k] for k in ['question','a','b','c','d'])):raise RuntimeError('P0 불일치')
    write(out/'prompt.json',{'name':name,**prompt,'system':m.SYSTEM_INSTRUCT,'example_user_message':rendered})
    write(out/'inference_config.json',settings)
    try:
        m.gpu_environment(settings,out);adapter=m.ModelAdapter(settings,out)
        adapter.model=PeftModel.from_pretrained(adapter.model,record['checkpoint'],local_files_only=True,is_trainable=False)
        probe=valid.iloc[0].to_dict();probe.pop('answer',None)
        inputs=adapter.encode(probe,training=False);input_tokens=int(inputs['input_ids'].shape[1]);adapter.generate(inputs);del inputs
        torch.cuda.synchronize()
        _,metrics=m.evaluate(adapter,valid,'valid',out)
        metrics.update({'checkpoint':record['checkpoint'],'train_resolution':640,'inference_resolution':672,
                        'loss_mode':cfg['loss_mode'],'warmup_samples':1,'first_input_tokens':input_tokens})
        write(out/'valid_metrics.json',metrics)
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        write(state_path,{'state':'completed','directory':str(out),'artifacts':files})
    except torch.cuda.OutOfMemoryError as exc:
        write(state_path,{'state':'blocked_oom','directory':str(out),'error':str(exc)})
    except BaseException as exc:
        write(state_path,{'state':'failed_or_interrupted','directory':str(out),'error':str(exc)});raise
    finally:
        with open(root/'CHANGELOG.md','a',encoding='utf-8') as f:f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M:%S')} {name}: {read(state_path)['state']}\n")
    summary(cfg)

if __name__=='__main__':
    cfg=read(sys.argv[1]);stage=sys.argv[2]
    if stage=='prepare':prepare(cfg)
    elif stage=='summary':summary(cfg)
    elif stage in cfg['prompt_order']:run(cfg,stage)
    else:raise ValueError(stage)

