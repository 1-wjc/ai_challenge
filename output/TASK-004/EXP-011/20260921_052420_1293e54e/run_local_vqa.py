
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def block_network():
    # HF flags alone do not cover all third-party code. Also reject Python socket access.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    os.environ.pop("HF_TOKEN", None)
    def guard(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise RuntimeError("학습·추론 프로세스의 네트워크 호출은 금지되어 있습니다.")
    sys.addaudithook(guard)


def read_table(path, columns):
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: 필수 컬럼 누락 {sorted(missing)}")
    if df.empty or df.id.eq("").any() or df.id.duplicated().any():
        raise ValueError(f"{path}: 빈 데이터, 빈 ID 또는 중복 ID")
    for col in columns:
        if df[col].str.strip().eq("").any():
            raise ValueError(f"{path}: {col}에 빈 값이 있습니다.")
    return df


def resolve_image(root, value):
    # CSV paths are local relative paths. Never fetch image URLs.
    value = str(value).replace("\\", "/")
    if "://" in value or Path(value).is_absolute() or ":" in value:
        raise ValueError(f"이미지는 DATA_DIR 내부 상대 경로여야 합니다: {value}")
    root = Path(root).resolve()
    p = (root / value).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise FileNotFoundError(f"이미지 경로 확인 필요: {p}")
    return p


def build_mc_prompt(row):
    return (f"{row['question']}\n"
            f"(a) {row['a']}\n(b) {row['b']}\n(c) {row['c']}\n(d) {row['d']}\n\n"
            "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.")


SYSTEM_INSTRUCT = ("You are a helpful visual question answering assistant. "
                   "Answer using exactly one letter among a, b, c, or d. No explanation.")


def parse_answer(raw):
    # Do not extract arbitrary letters from prose or default to a.
    s = str(raw).strip()
    s = re.sub(r"^(?:answer|정답)\s*[:：]\s*", "", s, flags=re.I)
    m = re.fullmatch(r"(?:\(([a-d])\)|([a-d]))[.。]?", s, flags=re.I)
    return (m.group(1) or m.group(2)).lower() if m else None


def prepare_data(cfg, out):
    import pandas as pd
    from PIL import Image, ImageOps
    root = Path(cfg["data_dir"])
    cols = ["id", "path", "question", "a", "b", "c", "d"]
    train = read_table(root / "train.csv", cols + ["answer"])
    test = read_table(root / "test.csv", cols)
    sample = read_table(root / "sample_submission.csv", ["id"])
    if set(sample.columns) != {"id", "answer"}:
        raise ValueError("sample_submission.csv 컬럼은 id, answer여야 합니다.")
    if not train.answer.isin(list("abcd")).all():
        raise ValueError("train answer는 공백 없는 소문자 a~d여야 합니다.")
    if set(sample.id) != set(test.id) or len(sample) != len(test):
        raise ValueError("sample_submission/test ID 집합·행 수가 다릅니다.")
    if set(train.id) & set(test.id):
        raise ValueError("train/test ID가 겹칩니다. 데이터 담당자 확인 필요.")
    for df in (train, test):
        for row in df.to_dict("records"):
            resolve_image(root, row["path"])
    if cfg["split_csv"]:
        split = read_table(cfg["split_csv"], ["id", "split"])
        if not split.split.isin(["train", "valid"]).all() or set(split.split) != {"train", "valid"}:
            raise ValueError("공통 split CSV는 id, split(train/valid) 형식이어야 합니다.")
        if not set(split.id) <= set(train.id):
            raise ValueError("분할 파일에 train.csv에 없는 ID가 있습니다.")
    else:
        n, nvalid = cfg["train_sample_n"], cfg["valid_n"]
        if not 0 < nvalid < n <= len(train):
            raise ValueError(f"{n}→{n-nvalid}/{nvalid} 분할 불가. 데이터 수={len(train)}. 설정을 명시적으로 수정하세요.")
        selected = train.sample(n=n, random_state=cfg["seed"]).reset_index(drop=True)
        split = pd.DataFrame({"id": selected.id,
                              "split": ["train"] * (n-nvalid) + ["valid"] * nvalid})
    split.to_csv(out / "split_manifest.csv", index=False)
    indexed = train.set_index("id", drop=False)
    tr = indexed.loc[split.loc[split.split.eq("train"), "id"]].reset_index(drop=True)
    va = indexed.loc[split.loc[split.split.eq("valid"), "id"]].reset_index(drop=True)
    # Exact decoded-pixel hashes detect re-encoded copies. Near-duplicates still need independent review.
    hashes = {}
    image_rows = []
    for label, df in [("train", tr), ("valid", va)]:
        for row in df.to_dict("records"):
            p = resolve_image(root, row["path"])
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                h = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                image_rows.append({"id":row["id"], "split":label, "path":row["path"],
                                   "pixel_sha256":h, "width":im.width, "height":im.height})
                hashes.setdefault(h, set()).add(label)
    pd.DataFrame(image_rows).to_csv(out / "image_audit.csv", index=False)
    if any(len(groups) > 1 for groups in hashes.values()):
        raise ValueError("학습/검증에 같은 픽셀의 이미지가 있습니다. image_audit.csv를 확인하고 공통 분할을 수정하세요.")
    info = {"train_csv_sha256":sha256(root / "train.csv"),
            "test_csv_sha256":sha256(root / "test.csv"),
            "sample_csv_sha256":sha256(root / "sample_submission.csv"),
            "split_sha256":sha256(out / "split_manifest.csv"),
            "train_n":len(tr), "valid_n":len(va), "test_n":len(test),
            "near_duplicate_review":"not_performed", "dev_used":False}
    write_json(out / "data_manifest.json", info)
    print("분할:", info, flush=True)
    return tr, va, test, sample, info


def make_submission(test, sample, predictions, path):
    import pandas as pd
    pred = pd.DataFrame(predictions)
    if pred.empty or pred.id.duplicated().any() or set(pred.id) != set(test.id):
        raise ValueError("예측 ID 중복·누락·추가가 있습니다.")
    if not pred.answer.isin(list("abcd")).all():
        raise ValueError("유효하지 않은 예측값이 있습니다.")
    submission = sample[["id"]].merge(pred[["id", "answer"]], on="id", how="left", validate="one_to_one")
    submission = submission.loc[:, list(sample.columns)]
    assert len(submission) == len(test) and submission.id.tolist() == sample.id.tolist()
    submission.to_csv(path, index=False, encoding="utf-8")
    reread = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert reread.equals(submission)
    return submission



def move_tensors(x, device, dtype):
    import torch
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype if x.is_floating_point() else x.dtype)
    if isinstance(x, dict):
        return {k:move_tensors(v, device, dtype) for k,v in x.items()}
    if isinstance(x, (tuple, list)):
        return [move_tensors(v, device, dtype) for v in x]
    return x


class ModelAdapter:
    def __init__(self, cfg, out):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        self.cfg, self.out = cfg, out
        self.kind = cfg["kind"]
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda:0")
        source = cfg["model_dir"]
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("이 비교 설정은 BF16을 지원하는 CUDA GPU가 필요합니다.")
        trust = self.kind == "minicpm"
        self.processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        # All options must be representable by one token for the explicit fallback.
        self.choice_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in "abcd"]
        if not all(len(x) == 1 for x in self.choice_ids) or len({x[0] for x in self.choice_ids}) != 4:
            raise ValueError("선지 토큰이 단일·고유 토큰이 아닙니다. 추론 코드 검토 필요.")
        self.choice_ids = [x[0] for x in self.choice_ids]
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype,
                    llm_int8_skip_modules=["visual", "vision_tower", "vision_model", "vpm",
                                           "resampler", "multi_modal_projector", "mlp1", "lm_head"])
        kwargs = dict(local_files_only=True, device_map={"":"cuda:0"},
                      torch_dtype=self.dtype, quantization_config=qconfig, attn_implementation="sdpa")
        if self.kind == "minicpm":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(source, trust_remote_code=True, **kwargs)
        else:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
        self.base = self.model
        self.model.eval()
        self.configure_processor()
        self.model.config.use_cache = False
        self.input_logged = False
        self.processor.save_pretrained(out / "processor")
        write_json(out / "model_config.json", self.base.config.to_dict())
        write_json(out / "processor_settings.json", {
            "kind":self.kind, "image_policy":cfg["image_policy"],
            "thinking":False, "choice_ids":self.choice_ids,
            "image_processor":self.processor.image_processor.to_dict()})

    def configure_processor(self):
        ip = self.processor.image_processor
        if self.kind == "qwen":
            # Qwen3.5 uses Qwen3VLProcessor / Qwen2VLImageProcessor.
            ip.size = {"shortest_edge":384*384, "longest_edge":384*384}
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = 384*384
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = 384*384
        elif self.kind == "internvl":
            ip.crop_to_patches = False
            ip.min_patches = 1
            ip.max_patches = 1
        elif self.kind == "gemma":
            ip.do_pan_and_scan = False
        # MiniCPM max_slice_nums=1 is passed per call without altering config invariants.

    def encode(self, row, training=False):
        import torch
        from PIL import Image, ImageOps
        p = resolve_image(self.cfg["data_dir"], row["path"])
        with Image.open(p) as f:
            image = ImageOps.exif_transpose(f).convert("RGB")
        prompt = build_mc_prompt(row)
        if self.kind == "minicpm":
            messages = [{"role":"system", "content":SYSTEM_INSTRUCT},
                        {"role":"user", "content":"(<image>./</image>)\n" + prompt}]
            if training:
                messages.append({"role":"assistant", "content":row["answer"]})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                       add_generation_prompt=not training, enable_thinking=False)
            inputs = dict(self.processor([text], [[image]], max_slice_nums=1,
                          return_tensors="pt", max_length=None))
            inputs.pop("image_sizes", None)
            inputs["input_ids"] = inputs["input_ids"].long()
        else:
            # Gemma doesn't accept an independent system role; place the same instruction in user text.
            messages = [] if self.kind == "gemma" else [
                {"role":"system", "content":[{"type":"text", "text":SYSTEM_INSTRUCT}]}]
            user_text = SYSTEM_INSTRUCT + "\n\n" + prompt if self.kind == "gemma" else prompt
            messages.append({"role":"user", "content":[{"type":"image"}, {"type":"text", "text":user_text}]})
            if training:
                messages.append({"role":"assistant", "content":[{"type":"text", "text":row["answer"]}]})
            text = self.processor.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=not training, enable_thinking=False)
            inputs = dict(self.processor(text=[text], images=[image], return_tensors="pt", padding=False,
                                         add_special_tokens=False))
        if inputs["input_ids"].shape[1] > self.cfg["max_input_tokens"]:
            raise ValueError(f"ID {row['id']}: 입력 길이 초과. 조용히 자르지 않습니다.")
        if training:
            labels = inputs["input_ids"].clone()
            # Full textual sequence supervision; ignore padding/media/role special tokens.
            labels[inputs["attention_mask"] == 0] = -100
            for sid in self.tokenizer.all_special_ids:
                if sid != self.tokenizer.eos_token_id:
                    labels[inputs["input_ids"] == sid] = -100
            for bound in inputs.get("image_bound", []):
                for start, stop in bound.tolist():
                    labels[:, start:stop] = -100
            inputs["labels"] = labels
        if not self.input_logged:
            def shape(x):
                if torch.is_tensor(x): return {"shape":list(x.shape),"dtype":str(x.dtype)}
                if isinstance(x,list): return [shape(y) for y in x]
                return str(type(x).__name__)
            write_json(self.out / "first_input_shapes.json", {k:shape(v) for k,v in inputs.items()})
            self.input_logged = True
        return move_tensors(inputs, self.device, self.dtype)

    def add_lora(self):
        import torch
        from peft import LoraConfig, get_peft_model
        # Equivalent k-bit preparation without temporarily doubling large frozen embeddings
        # in FP32. That transient allocation can exceed a 16 GiB card before being cast back.
        for name, p in self.model.named_parameters():
            p.requires_grad_(False)
            if "norm" in name.lower() and p.is_floating_point() and p.ndim == 1:
                p.data = p.data.to(torch.float32)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.model.enable_input_require_grads()
        suffixes = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
        excluded = {"visual","vision_tower","vision_model","vpm","resampler","multi_modal_projector","mlp1"}
        targets = [name for name, m in self.model.named_modules()
                   if name.split(".")[-1] in suffixes and not (set(name.split(".")) & excluded)
                   and hasattr(m, "weight")]
        if not targets:
            raise RuntimeError("언어 모듈 LoRA 대상이 없습니다.")
        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=targets,
                            bias="none", task_type=None if self.kind == "minicpm" else "CAUSAL_LM")
        self.model = get_peft_model(self.model, config)
        self.base = self.model.get_base_model()
        if self.kind == "minicpm":
            self.base.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.model.print_trainable_parameters()
        write_json(self.out / "lora_targets.json", targets)

    def loss(self, inputs):
        import torch
        if self.kind != "minicpm":
            return self.model(**inputs, use_cache=False).loss
        # MiniCPM has a custom data-dict forward; keep its vision/resampler frozen.
        with torch.no_grad():
            embeddings, _ = self.base.get_vllm_embedding(inputs)
        embeddings = embeddings.detach().requires_grad_(True)
        return self.base.llm(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"],
                             labels=inputs["labels"], use_cache=False).loss

    def generate(self, inputs, constrained=False):
        import torch
        kwargs = dict(max_new_tokens=1 if constrained else self.cfg["max_new_tokens"],
                      do_sample=False, num_beams=1, repetition_penalty=1.0, use_cache=True)
        if constrained:
            kwargs["prefix_allowed_tokens_fn"] = lambda batch_id, input_ids: self.choice_ids
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            if self.kind == "minicpm":
                embeddings, _ = self.base.get_vllm_embedding(inputs)
                ids = self.base.llm.generate(inputs_embeds=embeddings,
                        attention_mask=inputs["attention_mask"], pad_token_id=0,
                        eos_token_id=[self.tokenizer.convert_tokens_to_ids(t) for t in self.base.terminators],
                        **kwargs)
                # inputs_embeds-only generation returns generated token IDs, not the input text.
                generated = ids[0]
            else:
                ids = self.model.generate(**inputs, **kwargs)
                generated = ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def reload_adapter(self, directory):
        from peft import PeftModel
        # Unload adapter only, preserving the already loaded 4-bit base to avoid a second full model.
        base = self.model.unload()
        self.model = PeftModel.from_pretrained(base, str(directory), local_files_only=True, is_trainable=False)
        self.base = self.model.get_base_model()
        self.model.eval()



def evaluate(adapter, df, label, out, has_answers=True):
    import pandas as pd
    import torch
    adapter.model.eval()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    result = []
    # Stream predictions to disk to preserve completed rows on interruption.
    fields = ["id", "answer", "raw_output", "parse_failed", "fallback_output", "gold", "correct", "strict_correct"]
    with open(out / f"{label}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(df.to_dict("records")):
            # Explicitly remove gold before preparing every evaluation prompt.
            clean = {k:v for k,v in row.items() if k != "answer"}
            inputs = adapter.encode(clean, training=False)
            raw = adapter.generate(inputs)
            parsed = parse_answer(raw)
            answer, fallback = parsed, ""
            if answer is None:
                fallback = adapter.generate(inputs, constrained=True)
                answer = parse_answer(fallback)
                if answer is None:
                    raise RuntimeError(f"ID {row['id']}: 제한 생성도 a~d를 반환하지 않았습니다.")
            gold = row["answer"] if has_answers else ""
            rec = dict(id=row["id"], answer=answer, raw_output=raw, parse_failed=parsed is None,
                       fallback_output=fallback, gold=gold,
                       correct=(answer == gold) if has_answers else "",
                       strict_correct=(parsed == gold) if has_answers else "")
            result.append(rec)
            writer.writerow(rec)
            f.flush()
            del inputs
            if i % 25 == 0 or i+1 == len(df):
                print(f"{label}: {i+1}/{len(df)}", flush=True)
    seconds = time.perf_counter() - started
    metrics = {"n":len(result), "seconds":seconds, "seconds_per_sample":seconds/len(result),
               "parse_failure_rate":sum(r["parse_failed"] for r in result)/len(result),
               "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
               "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30}
    if has_answers:
        metrics.update(accuracy=sum(r["correct"] for r in result)/len(result),
                       strict_accuracy=sum(r["strict_correct"] for r in result)/len(result))
    write_json(out / f"{label}_metrics.json", metrics)
    print(label, metrics, flush=True)
    return result, metrics


def train_one_epoch(adapter, df, cfg, out):
    import torch
    from transformers import get_linear_schedule_with_warmup
    params = [p for p in adapter.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.01)
    updates = math.ceil(len(df)/cfg["gradient_accumulation"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates*0.03), updates)
    records = df.to_dict("records")
    random.Random(cfg["seed"]).shuffle(records)
    adapter.model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = []
    grad_checked = False
    # Correct normalization for the final incomplete accumulation group.
    for group_start in range(0, len(records), cfg["gradient_accumulation"]):
        group = records[group_start:group_start+cfg["gradient_accumulation"]]
        total_loss = 0.0
        for row in group:
            inputs = adapter.encode(row, training=True)
            with torch.autocast("cuda", dtype=adapter.dtype):
                loss = adapter.loss(inputs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"ID {row['id']}: loss가 NaN/Inf입니다.")
            raw_loss = float(loss.detach())
            (loss / len(group)).backward()
            total_loss += raw_loss
            del inputs, loss
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError("LoRA gradient가 없거나 NaN/Inf입니다.")
        if not grad_checked:
            if not any(bool(g.abs().max() > 0) for g in grads):
                raise RuntimeError("LoRA gradient가 모두 0입니다.")
            write_json(out / "backward_smoke.json", {"finite_loss":True,"nonzero_lora_gradient":True})
            grad_checked = True
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        rows.append({"update":len(rows)+1, "mean_loss":total_loss/len(group),
                     "grad_norm":grad_norm, "lr":scheduler.get_last_lr()[0]})
        print(f"train {len(rows)}/{updates}, loss={rows[-1]['mean_loss']:.4f}", flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "train_log.csv", index=False)
    metrics = {"epochs":1,"updates":len(rows),"seconds":time.perf_counter()-start,
               "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
               "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30}
    write_json(out / "train_metrics.json", metrics)
    del optimizer, scheduler, params, grads
    gc.collect()
    torch.cuda.empty_cache()
    return metrics



def main(cfg):
    block_network()
    import torch
    import numpy as np
    from importlib.metadata import version
    out = Path(cfg["run_dir"])
    write_json(out / "status.json", {"state":"started","gpu_executed":False})
    tr, va, test, sample, data_info = prepare_data(cfg, out)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다. 드라이버와 PyTorch CUDA 설치를 확인하세요.")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # A short actual CUDA computation catches missing Blackwell kernel support.
    x = torch.ones((32,32), device="cuda", dtype=torch.bfloat16)
    assert float((x @ x)[0,0]) == 32.0
    del x
    env = {"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
           "gpu":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0),
           "vram_gib":torch.cuda.get_device_properties(0).total_memory/2**30,
           "packages":{p:version(p) for p in ["transformers","peft","bitsandbytes","accelerate"]},
           "network":"blocked_in_training_and_inference_process",
           "determinism":"seed_fixed; nondeterministic-kernel warnings retained in log"}
    write_json(out / "environment.json", env)
    print(env, flush=True)
    adapter = ModelAdapter(cfg, out)
    _, baseline = evaluate(adapter, va, "valid_base", out)
    adapter.add_lora()
    training = train_one_epoch(adapter, tr, cfg, out)
    checkpoint = out / "adapter_epoch1"
    adapter.model.save_pretrained(checkpoint)
    adapter.processor.save_pretrained(checkpoint)
    gc.collect()
    torch.cuda.empty_cache()
    # Reload saved adapter bytes before producing validation and submission predictions.
    adapter.reload_adapter(checkpoint)
    _, tuned = evaluate(adapter, va, "valid_lora", out)
    # Keep selection explicit: this experiment submits the epoch-1 LoRA checkpoint.
    # Base and LoRA validation results are both reported; no Public-driven selection.
    predictions, inference = evaluate(adapter, test, "test", out, has_answers=False)
    submission = make_submission(test, sample, predictions, out / "submission.csv")
    summary = {"task_id":"TASK-004", "experiment_id":cfg["experiment_id"],
               "model":cfg["model_id"], "revision":cfg["revision"], "split":data_info,
               "baseline":baseline,"lora":tuned,"training":training,"test":inference,
               "submitted_checkpoint":"adapter_epoch1", "submission_rows":len(submission),
               "submission_sha256":sha256(out / "submission.csv"), "kaggle_uploaded":False,
               "public_score":None,
               "selection_note":"epoch-1 LoRA inference; model adoption awaits common-split review",
               "base_outperformed_lora":baseline["accuracy"] > tuned["accuracy"]}
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", {"state":"completed","gpu_executed":True,"kaggle_uploaded":False})
    print("완료:", out / "submission.csv", flush=True)
    print("Accuracy: base=", baseline["accuracy"], "LoRA=", tuned["accuracy"], flush=True)


if __name__ == "__main__":
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        main(cfg)
    except Exception as exc:
        write_json(Path(cfg["run_dir"]) / "status.json", {
            "state":"failed", "error_type":type(exc).__name__,"error":str(exc),
            "note":"실패한 실행의 부분 출력은 최종 제출물로 사용하지 마세요."})
        traceback.print_exc()
        sys.exit(1)
