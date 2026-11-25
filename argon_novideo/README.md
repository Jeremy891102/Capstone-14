# HD-EPIC Argon 無影片 batching：執行前方案

狀態：2026-11-25 離線準備完成；本聊天室尚無付費呼叫。使用者要求先確認方案與預算，且每個階段完成後回報。

## 已凍結的輸入

`frozen/k{1,4,8,16,32}.jsonl` 分別有 2,336／584／292／146／73 次呼叫，每個 k 都覆蓋同一批 73 部影片、2,336 個 localization targets。k=1 從原始 3,212 個全題型請求篩出共同題目；k=4/8/16/32 保持原始巢狀順序。

`prepare.py` 從每個 baseline `full.py` 擷取原始 `snapshot_for` 函式，還原實際 payload 後，只移除影片 inlineData。所有文字逐字保留，包括「Video head_cam; this clip starts at 0 seconds.」、system、user、題目與選項。generationConfig 保持 MEDIUM、32768、JSON；不設定 temperature。題目內無影片的敘述也不改寫。

來源與檔案 SHA256 見 `frozen/manifest.json`。答案只放 `evaluation_only.jsonl`，不包含在 API body。

## 第一關：k=1 subset10 對照

`frozen/k1_subset10.jsonl` 的 320 個完整 body 與 11/12 無影片 single 條件逐筆完全相同。既有 320 個成功回應亦已重新驗證：79/320，24.6875%。證據見 `readiness_audit.json`，原始來源：

`/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/research_next_step_20261111/execution_20261112/`

執行器 `trial.py` 沿用上述實驗的 harness，完整副本保存在 `code/historical_runner.py`，並核對其 hash。序列執行、無自動重試；既有成功呼叫可恢復，失敗或不確定狀態須先檢查。API 原始回應與 thinking tokens 均保存於新的 `trial_results/`。完成後輸出 accuracy、相對歷史差異與逐題答案一致率；差異超過 5 pp 標記 investigate。任何結果都不自動啟動下一關。

離線檢查（已執行，不讀取 API key）：

```sh
python argon_novideo/trial.py
python -m unittest discover -s argon_novideo -p 'test_*.py'
```

待預算同意後才執行：

```sh
python argon_novideo/trial.py --execute --max-estimated-usd 2
```

工作目錄：`/Users/jeremy/Documents/ChatGPT/Capstone`。使用既有環境及 Python 標準函式庫，不安裝或升級套件。API key 只在執行時從環境或既有 .env 讀入，不寫入結果。

## 費用與批准範圍

Google 官方價格頁：https://cloud.google.com/model-garden/pricing （2026-11-25 查閱）。Global 標準線上推論在 11/30 前標示 input $2/M、cache read $0.10/M、output response + reasoning $10/M；優惠以 50% 帳單抵扣提供。未計抵扣時以 $4/$0.20/$20 估算；未查本帳戶帳單或優惠資格。這裡的 batch size 指一次一般線上呼叫內題目數，沒有使用折扣 Batch API。

歷史相同 320 次 single 呼叫：155,115 input、5,578 answer、5,208 thinking tokens，優惠價估算 $0.41809；以未抵扣費率為 $0.83618。新回應長度可能不同。

建議本次僅批准第一關：320 次，優惠價估算控制額 $2，等同未抵扣費率 $4 的控制額。這是本地使用量估算停止線，並非雲端硬性帳單上限；未知帳單的失敗呼叫會停下且不自動重試。

## 下一關與正式實驗（尚未執行）

第一關完成並回報後，再執行 `frozen/cost_pilot.jsonl`：每個 k=4/8/16/32 各 5 次，共 20 次；選取凍結 k32 序列中索引 0/18/36/54/72 的五部影片，每部取該 k 的第一個完整 batch。回報各 k 的 input/answer/thinking tokens、估計單次費用及全跑推算。這一關尚未建立批次答案執行器，會在第一關審查後準備。

正式 k=4/8/16/32 共 1,095 次；成功的費用試跑呼叫將計入，不重複送出。交接估算 $10–20、建議上限 $30，仍待實測取代。全資料 k=1 無影片結果是否可完全重用，尚待定位並核對 73 部影片的原始回應；不可把 subset10 的 24.6875% 當成全體 k=1。

之後才做每個 k 的有／無影片準確率、配對影片增益及指定 contrasts，以 73 部影片 bootstrap 10,000 次。原始 baseline run 資料夾日期為 10/26–10/28；11/09–11/10 是彙整報告日期，實際呼叫時間需由回應紀錄確認。最終分析須註明非同時期對照。

## 驗證與範圍

兩個單元測試驗證：只移除 media、原物件不被改寫、文字保留、媒體數量異常拒絕。真實資料驗證另涵蓋 cohort、唯一題目、target 一致、巢狀順序、320 個歷史完整 body 一致，以及歷史 raw response／usage 重算。

未套用 Desktop patch、未改動原始結果、未啟動 VM/GPU 工作，未 commit 或 push。本工作目錄的 git 尚無初始 commit，另有其他工作的未追蹤檔，故此次檔案保留為獨立未提交目錄。

## 第一關完成更新

2026-11-25：使用者明確同意後，320 次試跑完成。新準確率26.25%、歷史24.6875%、差+1.5625 pp；逐題一致55.9375%。優惠估算費用 $0.38729，0失敗。詳見 trial_results/report.md。原先「尚無付費呼叫」是執行前狀態，現由此更新取代。下一關未啟動。

完整歷史 k1 亦已找到並核對：2336 次完整 body 與原始回應全通過，620/2336=26.5411%；詳見 full_k1_historical_audit.json。

## 第二關完成更新

20次費用試跑完成，詳見 cost_pilot_results/report.md。優惠估算$0.205064；完整1095次外推$6.9555568，餘1075次$6.7504928。正式runner full_run.py已離線驗證，會重用20次pilot，正式呼叫尚未啟動，等待使用者預算批准。

## 正式實驗完成更新

正式實驗與分析全部完成，完整交付在 `../results/hd_epic_argon_novideo_batching_20261125/README.md`。先前「等待預算批准／尚未啟動」為歷史準備狀態。1095個成功批次結果完整，1次HTTP500覆核後補送；批次已知成功估算$6.793754。本次無影片k8−k1 +5.74pp，但影片增益k8−k1僅+0.68pp，CI[-2.18,3.55]。
