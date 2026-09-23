# TASK-006 loss 비교

학습 640² / 추론 672². 기존 384 학습 점수와 직접 비교하지 않음.

  loss_mode    status  train_resolution  inference_resolution   n  correct_n  accuracy  parse_failure_rate  fallback_usage_rate    seconds  seconds_per_sample  peak_allocated_gib  peak_reserved_gib                                                                                                                                checkpoint  training_seconds  optimizer_updates  training_peak_allocated_gib  reused_baseline  warmup_samples  accuracy_pct  gain  loss  delta_pp
  full_text completed               640                   672 500        459     0.918                 0.0                  0.0 391.805877            0.783612            7.652091          11.333984   C:\Users\SSAFY\Desktop\AI2_Challenge\output\TASK-006-loss\LOSS-51ca09225d304a9a\train_full_text_20260922_143828_bff2578f\adapter_epoch1       2532.541633                250                     9.904253            False               1          91.8   NaN   NaN       NaN
answer_only completed               640                   672 500        470     0.940                 0.0                  0.0 441.341582            0.882683            7.652091          11.333984 C:\Users\SSAFY\Desktop\AI2_Challenge\output\TASK-006-loss\LOSS-51ca09225d304a9a\train_answer_only_20260922_152814_5e73bca1\adapter_epoch1       2658.919027                250                     9.904253            False               1          94.0  16.0   5.0       2.2

독립 검토/채택 미완료.