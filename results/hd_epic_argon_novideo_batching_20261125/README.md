# HD-EPIC Argon 無影片 batching（2026-11-25）

狀態：完成並通過原始回應、題目、用量及配對稽核。所有新結果都在本目錄，沒有改動既有 baseline 或套用 Desktop patch。

## 主要發現

無影片曲線也隨 k 上升。k8−k1 的無影片提升為 +5.74 pp [3.13, 8.48]；同一對比的有影片提升為 +6.42 pp [4.41, 8.39]。影片增益差（有影片提升減無影片提升）僅 +0.68 pp [−2.18, 3.55]：沒有明確證據支持 k=8 相對 k=1 有額外影片增益。這與文字 batching 能提升作答的解釋一致；不能宣稱所有提升都由文字造成，或排除仍有影片效果。

k32−k8：無影片 +5.52 pp [3.08, 8.05]；有影片 −5.18 pp [−7.45, −2.95]；影片增益差 −10.70 pp [−13.74, −7.71]。較大 k 的影片條件相對無影片條件明顯失去優勢。這描述的是整體推論協定差異，不單獨識別視覺注意力或其他內部機制。

數字來源：`accuracy_summary.csv`、`contrasts.csv`，底層配對來源 `paired_per_field.csv`。下列全部區間為 95% CI，百分比與百分點不可混用。

## 每個 k 的結果

- k=1：有影片 1402/2336 = 60.02% [56.81, 63.23]；無影片 620/2336 = 26.54% [24.70, 28.30]；影片增益 33.48 pp [30.22, 36.69]。
- k=4：有影片 1518/2336 = 64.98% [61.86, 68.19]；無影片 723/2336 = 30.95% [28.94, 32.92]；影片增益 34.03 pp [30.91, 37.24]。
- k=8：有影片 1552/2336 = 66.44% [62.97, 69.91]；無影片 754/2336 = 32.28% [29.71, 34.85]；影片增益 34.16 pp [30.74, 37.59]。
- k=16：有影片 1525/2336 = 65.28% [62.11, 68.45]；無影片 826/2336 = 35.36% [32.83, 37.89]；影片增益 29.92 pp [26.97, 32.92]。
- k=32：有影片 1431/2336 = 61.26% [58.18, 64.38]；無影片 883/2336 = 37.80% [34.85, 40.75]；影片增益 23.46 pp [20.51, 26.41]。

## 設定與時間差

模型 gemini-4-argon，Google Vertex Global 直接一般線上呼叫；沒有使用 Batch API 折扣端點。MEDIUM、maxOutputTokens=32768、JSON；不自行設定 temperature、seed。無 agentic 工具、無影片或音訊輸入。凍結輸入還原自五個 baseline full.py 的 snapshot_for，僅移除 inlineData，逐字保留 system/user、Video head_cam 標籤、題目、選項、時間文字、題目組與順序。原 baseline 影片為 2 FPS、480px 高、無音訊、原始指定時間窗口。

五種 k 都為73部影片，每部32題，共2336個共同 localization targets。原全題型 k1 的3212題已篩出共同2336題。每個 k 的 call_id 與逐call題目順序已和 baseline CSV 核對一致。

有影片來源：`/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/baseline_complete_20261109/per_field.csv`，以及凍結 manifest 中的五個 Vertex run（資料夾日期10/26–10/28，報告日期11/09–11/10不等於實際推論日期）。

無影片 k1 主曲線重用11/12全73影片歷史結果，2336個 body 與新凍結k1逐筆相等，raw response重新解析通過；不把本次subset10試跑當成全體k1。來源：`/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/argon_reference_novideo_full73_20261112/`。k4/8/16/32為本次2026-11-25新增結果。因此有影片baseline、完整無影片k1、本次其他k不是同時期收集。

另做的當期k1 subset10試跑：84/320=26.25%，歷史79/320=24.69%，差+1.56pp；答案一致179/320=55.94%。這未超過事前約5pp門檻，但不能據此排除供應商或採樣漂移。見`k1_subset10_trial_report.md`與audit。

## 統計方法

以影片為抽樣單位，73部影片有放回抽樣10000次，seed20261027，percentile 2.5%/97.5%。每個條件與k均使用同一組抽樣索引；所有影片增益與contrast直接對逐片差值做配對bootstrap。每片均32題，因此等影片權重等同題目權重。精確重現原有影片k8−k1的+6.4212pp與CI[4.4092,8.3904]。未做多重比較校正。CI反映影片抽樣變異，不涵蓋非同時期收集等系統性誤差。

## 呼叫、費用與失敗紀錄

k4/8/16/32接受1095個成功結果（584/292/146/73），含20個逐位元相同重用的費用試跑；另外保留2336個歷史k1成功回應。正式批次成功用量估算US$6.793754，本次另做k1 subset10估算US$0.387290，新增工作已知成功費用共US$7.181044。歷史k1估算US$2.911028不計入本次新增費用。

費率按2026-11-25查閱的Google官方Global優惠價input $2/M、cache read $0.10/M、output(response+thinking) $10/M；優惠透過50%帳單抵扣提供。所有本次新增工作的未抵扣等值約US$14.362088。這些是用量估算，未核實實際帳單或優惠資格。官方來源：https://cloud.google.com/model-garden/pricing 。

途中1次HTTP500 INTERNAL，無答案、無usage。runner停止，人工檢查後封存原檔與hash，再補送相同失敗請求；成功結果沒有重跑。合計批次HTTP嘗試1096次，1次失敗可能帳單未知，未包含在上述已知成功用量費用。原始檔在`attempt_history/reviewed_http500_001/`，人工覆核見`failure_review_001.json`。原US$10優惠估算控制額中預留US$0.40給未知失敗費用，續跑成功用量停止線降到US$9.60。原始指令、狀態與續跑過程均保存在`full_run.log`。

使用失敗處理官方說明：https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/api-errors 。沒有啟動選做的有影片漂移重跑、+3參考題或GPU工作。未升級任何既有環境套件。

## 檔案與重算

- `per_field.csv`：五種k全部無影片逐題預測，11680列，與baseline主要欄位相同。
- `baseline_per_field.csv`：同一共同題目的有影片匯出。
- `paired_per_field.csv`：逐題有／無影片配對。
- `per_call.csv`：3431個接受呼叫的用量與來源，含歷史k1。
- `accuracy_summary.csv`、`contrasts.csv`、`per_video.csv`：摘要與統計來源。
- `accuracy_and_video_gain.png`／`.svg`：兩條準確率曲線及配對影片增益。
- `calls/`：本次1095個原始回應。`historical_k1_calls/`：2336個完整歷史回應副本。
- `inputs/`：凍結請求與評分答案，答案不在API body中。
- `summary.json`、`completion_audit.json`、`raw_response_hashes.json`：稽核、來源與hash。
- `code/`：本次執行、分析與測試程式副本。

在原工作目錄離線重算（不呼叫API、不讀API key）：

```sh
/Users/jeremy/Documents/Codex/2026-10-06/git/work/baseline-report-env/bin/python argon_novideo/analyze.py
/Users/jeremy/Documents/Codex/2026-10-06/git/work/baseline-report-env/bin/python -m unittest discover -s argon_novideo -p "test_*.py"
```

該既有獨立報告環境numpy2.5.4/matplotlib3.11.2可用。預設Anaconda matplotlib存在NumPy ABI不相容，因此沒有改動它；API執行器只用Python標準庫，不受繪圖環境影響。程式副本保留原絕對來源路徑，移機需改來源路徑。7個測試通過。完整執行過程未commit或push。
