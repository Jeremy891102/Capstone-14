# CC4D Argon 單步 oracle：20 步試跑

2026-11-25 完成。這是流程與成本試跑，尚未執行正式 951 步；分層挑選的 20 步不作母體成效推論。

## 已驗證

- 951 reports、1,901 fields、64 部錄影；和舊 oracle 的 report ID、影片區間逐筆相同。
- 分組：做對 411、Order-only 104、Order＋其他 94、無 Order 的錯誤 342。
- 四組各取 5 步，依組內片段長度五分位選擇中央附近樣本，並要求 20 部不同錄影；選取完全不讀模型輸出。樣本固定於 `data/cc4d_oracle_single_pilot20/`，ID 與 hashes 見 `evidence/readiness_audit.json`。
- 20/20 API 成功，20/20 嚴格 JSON／型別／keys 檢查通過，40 fields 無解析錯誤、缺答或非法答案；無失敗、無重試。
- input 87,839；answer output 475；thinking 1,802；cache 0 tokens。
- 實際送出的 generationConfig 逐筆核對：temperature 0、maxOutputTokens 8192、JSON；未指定 thinking level。模型回報版本 gemini-4-argon。

## 費用

Google 官方 https://cloud.google.com/model-garden/pricing （2026-11-25 查閱）列 Global 標準線上推論的 input $4/M、cache $0.20/M、output＋reasoning $20/M；11/30 前的優惠價透過 50% credits 提供，本帳戶資格及帳單未核實。

用回傳 usage 計算，試跑抵扣前 $0.396896；若取得抵扣，淨額約 $0.198448。這是使用量估算，不是已查核帳單。

951 步總費用：按分組加權約 $18.17 抵扣前；以 input tokens 對片段長度擬合，再校正全體片段長度，約 $19.48。若取得優惠，約 $9.08–9.74。20 次樣本不足以精確預測長尾及服務端 token 漂移，建議正式階段總預算採抵扣前 $30，保留原先建議 $40 作討論上限。試跑的 20 步可以重用，剩餘 931 步不能再送這 20 步。

試跑本地停止線是累計抵扣前 $2，達線後不再送下一次；單次可能越過停止線，並非雲端硬性帳單上限。序列執行，max_attempts=1，遇失敗或 usage 不明停止。

## Prompt 與控制

採 patch 的單步 context、A–E、±5 秒片段，但本次 user template 不使用共用 `$answer_format`，避免它繼續插入非空清單範例 `["A", "C"]`。改以中性型別規則說明 true/false、A–E list、[]、not_visible。只改本實驗模板，共用 prompting/provider 與既有結果皆未更動。完整渲染範例見 `full_prompt_example.txt`。

影片由 VM 既有 360p 檔案連結，使用原 provider 的本機裁切、scale height480、2fps、移除音訊設定，不使用 GPU。

## Baseline 核對與待辦

從原始 rows、GT、abstention 與 correctness 重建 boolean 預測，限制在相同 951 步：正向 k8 BA 69.3883%，正向 kall 68.0082%。952→951 差異是 cc4d-18_101 step09_correct：target=true 但時間 -1/-1，沒有可用 oracle 片段。詳見 readiness_audit。

目前尚未定位 Argon 原始 full-video field_scores，所以多選答案不能由 rows 還原；正式 A–E exact-match 對照前必須補齊。既有已驗證 oracle-order-scoring script 的 main 還硬編碼驗證 Qwen 成績，Argon 需另建入口、重用其統計函式，不可直接照抄 CLI。正式分析需含配對 recording-bootstrap CI、四組分析、排除 Order 子集與 relabel 敏感度分析。pilot 的精度數字僅保存於原始 evaluation，不能視為正式效果。

## 證據與重現

本機 worktree：`/Users/jeremy/Documents/ChatGPT/Capstone/argon_oracle`，branch `codex/argon-oracle-single-20261125`，base `4bd16a1`，新程式尚未 commit/push。來源 hashes 見 `evidence/code_provenance.json`。

VM 隔離目錄：`/home/nyu-data/argon_oracle_20261125`。

- `runs/argon_oracle_pilot20/`：frozen snapshot、逐次 checkpoints、responses、evaluations。
- `runs/argon_oracle_pilot20/api_raw/`：20 份原始 API JSON、20 份 wire payload（媒體以 base64 SHA256／長度替代，無 credential）。
- 本目錄 `per_call.csv`、`summary.json`：逐次用量、延遲、回答、費用。
- `pilot_live.log`、`pilot_evaluation.log`：執行與離線評分紀錄。

VM 執行命令（已完成，請勿重送）：

```sh
cd /home/nyu-data/argon_oracle_20261125
PYTHONPATH=src ~/envs/capstone-qwen8b/bin/python scripts/run_argon_oracle_pilot.py --execute
PYTHONPATH=src ~/envs/capstone-qwen8b/bin/python -m video_report.cli evaluate --run runs/argon_oracle_pilot20 --ground-truth data/cc4d_oracle_single_pilot20/ground_truth.jsonl
PYTHONPATH=src ~/envs/capstone-qwen8b/bin/python scripts/summarize_argon_oracle_pilot.py
```

驗證：既有套件 310 passed；加入停跑測試後全套 312 passed、1 skipped（VM 缺 google.genai，Developer API adapter）、1 deselected（live）。之後再加的費用停止／usage 不明／不保存 key 測試，pilot 專項合計 4 passed。mock 20 次執行與評分通過。試跑原始回覆已複製回本機，並再次核對 20 單次 attempts、20 raw API JSON 與實際 generationConfig。
