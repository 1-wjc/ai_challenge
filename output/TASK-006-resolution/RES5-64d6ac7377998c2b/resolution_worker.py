
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
    m=a.merge(b,on='id',validate='one_to_one',suffixes=('_384','_new'))
    if len(m)!=len(a) or len(m)!=len(b) or not (m.gold_384==m.gold_new).all():
        raise RuntimeError('비교 ID/정답이 동일하지 않습니다.')
    if not (m.group_id_384==m.group_id_new).all(): raise RuntimeError('이미지 그룹 불일치')
    old=m.answer_384==m.gold_384;new=m.answer_new==m.gold_new
    m['transition']=['gain' if y and not x else 'loss' if x and not y else 'same_correct' if x else 'same_wrong' for x,y in zip(old,new)]
    return m,{'gain':int((~old & new).sum()),'loss':int((old & ~new).sum()),
              'delta_pp':100*float(new.mean()-old.mean())}


def summarize(cfg):
    import pandas as pd
    root=Path(cfg['output_dir']);rows=[];ref=None
    r=verified_status(root,384)
    if r['state']=='completed':ref=pd.read_csv(Path(r['directory'])/'valid_predictions.csv',keep_default_na=False)
    types=[]
    for size in cfg['resolutions']:
        rec=verified_status(root,size);row={'resolution':size,'pixel_budget':size*size,'status':rec['state']}
        if rec['state']=='completed':
            d=Path(rec['directory']);metrics=read(d/'valid_metrics.json');row.update(metrics)
            row['accuracy_pct']=100*metrics['accuracy']
            row['parse_failure_pct']=100*metrics['parse_failure_rate']
            pred=pd.read_csv(d/'valid_predictions.csv',keep_default_na=False)
            if ref is not None:
                joined,stats=paired(ref,pred);row.update(stats)
                joined.to_csv(root/f'paired_384_vs_{size}.csv',index=False,encoding='utf-8-sig')
                joined[joined.transition.isin(['gain','loss'])].to_csv(root/f'changed_384_vs_{size}.csv',index=False,encoding='utf-8-sig')
            for name,g in pred.groupby('question_type'):
                types.append({'resolution':size,'question_type':name,'n':len(g),'correct_n':int((g.answer==g.gold).sum()),'accuracy':float((g.answer==g.gold).mean())})
            row['result_directory']=str(d)
        else:row['error']=rec.get('error','')
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(root/'resolution_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(types,columns=['resolution','question_type','n','correct_n','accuracy']).to_csv(root/'type_comparison.csv',index=False,encoding='utf-8-sig')
    complete=[x['resolution'] for x in rows if x['status']=='completed']
    write(root/'summary.json',{'completed':complete,'total_conditions':len(rows),'adoption':'not_selected',
        'baseline_run':cfg['baseline_dir'],'test_used':False,'holdout_evaluated':False,'review_02':'pending','results':rows})
    (root/'PROJECT_STATUS_update.md').write_text('# TASK-006 추론 해상도 비교\n\n완료 조건: '+str(complete)+'\n\n동일 저장 LoRA/고정 검증셋. 재학습·test·holdout 평가 없음. 채택/독립 검토 미완료.\n\n결과: '+str(root/'resolution_comparison.csv'),encoding='utf-8')
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
    _,valid,info=module.load_split(original,base)
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
            'valid_n':len(valid),'previous_predictions':str(Path(evaluation['directory'])/'valid_lora_predictions.csv'),
            'model_file_stats':{name:[(Path(original['model_dir'])/name).stat().st_size,(Path(original['model_dir'])/name).stat().st_mtime_ns] for name in assets['files']}}
    frozen_path=root/'frozen_inputs.json'
    if frozen_path.exists() and read(frozen_path)!=frozen:raise RuntimeError('기준 입력 변경. 새 SESSION_TAG로 실행하세요.')
    write(frozen_path,frozen)
    print('준비 완료. 저장 LoRA:',frozen['checkpoint'],'검증:',len(valid),flush=True)
    summarize(cfg)


def run_resolution(cfg,size):
    import torch,pandas as pd
    from peft import PeftModel
    root=Path(cfg['output_dir']);base=Path(cfg['baseline_dir']);frozen=read(root/'frozen_inputs.json')
    module,original=load_baseline(cfg)
    if digest(base/'run_config.json')!=frozen['baseline_config_sha256']:raise RuntimeError('baseline 설정 변경')
    for name,sha in frozen['baseline_records'].items():
        if digest(base/(name+'_complete.json'))!=sha:raise RuntimeError('baseline 완료 기록 변경')
    # Revalidate checkpoint, split and current image bytes before reuse/evaluation.
    module.completed(base,'train')
    _,valid,info=module.load_split(original,base)
    if digest(root/'fixed_valid_ids.csv')!=frozen['valid_ids_sha256']:raise RuntimeError('고정 검증 ID 파일 변경')
    ids=pd.read_csv(root/'fixed_valid_ids.csv',dtype=str)
    if ids.id.tolist()!=valid.id.tolist():raise RuntimeError('검증 ID 순서 변경')
    for name,stat in frozen['model_file_stats'].items():
        f=Path(original['model_dir'])/name
        if [f.stat().st_size,f.stat().st_mtime_ns]!=stat:raise RuntimeError('모델 파일 변경: 준비 셀 재실행 필요')
    previous=verified_status(root,size)
    if previous['state']=='completed' or (previous['state']=='blocked_oom' and not cfg['retry_oom']):
        print('기존 상태 재사용:',size,previous['state'],flush=True);summarize(cfg);return
    out=root/f'res_{size}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}';out.mkdir()
    settings=dict(original);settings['pixel_budget']=size*size;settings['image_policy']=f'inference pixel budget {size}²; original aspect ratio'
    settings['run_dir']=str(out)
    write(out/'inference_config.json',settings)
    status_path=root/f'res_{size}_status.json'
    write(status_path,{'state':'running','directory':str(out)})
    try:
        module.gpu_environment(settings,out)
        adapter=module.ModelAdapter(settings,out)
        adapter.model=PeftModel.from_pretrained(adapter.model,frozen['checkpoint'],local_files_only=True,is_trainable=False)
        adapter.model.eval()
        # No adapter.add_lora / optimizer / training. Warm up same first validation input.
        probe=valid.iloc[0].to_dict();probe.pop('answer',None)
        adapter.generate(adapter.encode(probe,training=False));torch.cuda.synchronize()
        _,metrics=module.evaluate(adapter,valid,'valid',out)
        metrics.update({'checkpoint':frozen['checkpoint'],'train_pixel_budget':original['pixel_budget'],
                        'inference_pixel_budget':size*size,'warmup_samples':1,'timing':'validation loop only, excludes load/warmup'})
        write(out/'valid_metrics.json',metrics)
        if size==384:
            old=pd.read_csv(frozen['previous_predictions'],keep_default_na=False)
            new=pd.read_csv(out/'valid_predictions.csv',keep_default_na=False)
            m,_=paired(old,new)
            write(out/'baseline_reproduction.json',{'n':len(m),'same_predictions':bool((m.answer_384==m.answer_new).all()),
                'same_raw_outputs':bool((m.raw_output_384==m.raw_output_new).all()),
                'changed_prediction_count':int((m.answer_384!=m.answer_new).sum()),
                'note':'timing rerun with warmup; discrepancy requires review'})
        artifacts={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
        write(status_path,{'state':'completed','directory':str(out),'artifacts':artifacts})
    except torch.cuda.OutOfMemoryError as exc:
        # No smaller resolution, changed quantization, or partial accuracy substituted.
        write(status_path,{'state':'blocked_oom','directory':str(out),'error':str(exc),
            'partial_predictions_are_not_metrics':True})
        print('OOM으로 해당 조건 차단:',size,flush=True)
    except BaseException as exc:
        write(status_path,{'state':'failed_or_interrupted','directory':str(out),'error':str(exc)})
        summarize(cfg);raise
    finally:
        with open(root/'CHANGELOG.md','a',encoding='utf-8') as f:
            f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M:%S')} / {size}² / {read(status_path)['state']} / {out}\n")
    summarize(cfg)


if __name__=='__main__':
    config=read(sys.argv[1]);stage=sys.argv[2]
    if stage=='prepare':prepare(config)
    elif stage=='summary':summarize(config)
    else:
        size=int(stage)
        if size not in config['resolutions']:raise ValueError('설정에 없는 해상도')
        run_resolution(config,size)

