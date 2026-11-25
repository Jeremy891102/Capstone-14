# CC4D Argon 單步 oracle：正式結果（2026-11-25）

## 結論

951 步主實驗完成。**原標籤下，oracle 與完整影片 k=8 的平衡準確率沒有明確差異；排除所有含 Order 錯誤的步驟後，oracle 的平衡準確率較高。** 這不是「oracle 完全沒用」，也不是「已證明瓶頸就在定位」。

非 Order 子集的收益主要來自較少把做對的步驟誤判成做錯；對真正做錯、且不含 Order 的步驟，oracle 的偵錯率沒有提高。因此這個結果更適合解讀為**局部證據、prompt 與回答傾向的共同改變，加上裁切時的順序上下文損失**；要識別裁切本身的效果，仍需固定單步 prompt、題目分組及文字 context 的完整影片控制組。

## 執行與預算

- 64 部錄影、951 步、1,901 個 fields；20 筆 pilot 完整重用，另補931筆，沒有重送。
- 951/951 呼叫成功，951/951 嚴格 JSON／key／型別檢查通過；無失敗、無重試。not_visible 是合法棄答，沒有從分母排除。
- 回傳模型版本全部為 `gemini-4-argon`。所有實際 generationConfig 均核對為 temperature0、maxOutputTokens8192、JSON，未指定 thinking level。
- input 4,357,186、answer 22,741、thinking 96,116、cached input 3,384 tokens（cache 是 input 的子集，不另加總）。
- **抵扣前使用量估算 US$19.793025**，低於已批准總預算US$30。若取得50%優惠抵扣，約US$9.896512；帳戶資格及帳單未核實。
- 計價來源：[Google 官方價格頁](https://cloud.google.com/model-garden/pricing)，2026-11-25核對；Global標準線上 input$4/M、cache$0.20/M、output＋thinking$20/M，優惠透過credits處理。沒有使用折扣Batch API。
- Pilot：美東11:43–11:45；正式931步：美東11:52–12:34（原始UTC時間見execution.json）。

## 主要比較：同一批951步，原始correct標籤

平衡準確率是「做對步驟上的準確率」和「做錯步驟上的準確率」的平均；所有結果以錄影為叢集，2,000次bootstrap、seed20261124，差異直接使用相同抽樣計算。

- Oracle：69.96%［95% CI 67.04, 72.82］。
- 完整影片正向k=8：69.39%［95% CI 65.58, 73.28］。
- 完整影片正向整份report：68.01%［95% CI 64.24, 71.73］。
- Oracle − k=8：**+0.57 pp［95% CI -2.34, +3.52］**。
- Oracle − 整份report：+1.95 pp［95% CI -1.93, +6.25］。

差值區間包含零，表示目前無法清楚區分兩個條件的整體成績，不能宣稱等效。原始accuracy分別為oracle68.45%、k8 69.30%、整份report68.45%；資料不平衡，因此主要看balanced accuracy。

## Order分析

**排除所有含Order的步驟**：保留411個做對＋342個不含Order的做錯步驟，共753步。

- Oracle：72.53%［95% CI 69.40, 75.77］。
- k=8：68.37%［95% CI 64.54, 72.34］；整份report：66.28%［95% CI 62.37, 70.20］。
- Oracle − k=8：**+4.16 pp［95% CI +0.76, +7.37］**。
- Oracle − 整份report：+6.25 pp［95% CI +2.15, +10.46］。

這是事先規劃的次要分析；CI未作多重比較校正。

**Order-only重標為做對**是敏感度分析，不是另行人工驗證的片段真值：oracle BA為71.71%［95% CI 68.75, 74.71］，相對k8差+4.45 pp［95% CI +1.34, +7.34］。短片可能保留部分相鄰事件，不能斷言所有Order錯誤都絕對不可見。

四組原標籤準確率（oracle／k8／整份report）：

- done correctly，n=411：81.02%／70.07%／64.72%。
- Order only，n=104：31.73%／60.58%／67.31%。
- Order + other，n=94：70.21%／85.11%／88.30%。
- no Order，n=342：64.04%／66.67%／67.84%。

Oracle在做對步驟上由k8的70.07%升至81.02%，但無Order的做錯步驟由66.67%降至64.04%；Order-only由60.58%降至31.73%。所以「給對片段後更會抓錯」不是這組數據支持的直接結論。Order組下降與上下文損失相容，但prompt、上下文及題目分組也同時不同，仍非單因素因果證明。

## 回答傾向與錯誤類型

Oracle回答做對58.04%（原標籤真實做對比例411/951=43.22%）；做對召回81.02%、做錯召回58.89%。correct題4筆回答not_visible，無非法／缺失boolean。

A–E完全比對使用950題：cc4d-26_39__step08沒有error_types target，因此正確題仍保留在951步中，多選分母另為950。不是因模型失敗而刪題。

- Oracle A–E exact match：**60.00%［95% CI 57.07, 62.97］**（570/950）。
- 永遠回答[]：54.21%［95% CI 51.01, 57.14］（515/950）。
- 配對差異：+5.79 pp［95% CI +2.44, +9.31］。
- Oracle回答[]為58.11%，多選not_visible為4/950；非法／缺失答案仍計入分母。

各類型recall／false positive rate（使用A–E原標籤，非Q1的泛稱「做錯」）：
- Preparation：56.93%［95% CI 45.65, 68.26］／6.52%［95% CI 4.95, 8.39］。
- Measurement：64.22%［95% CI 54.62, 73.40］／8.20%［95% CI 6.26, 10.23］。
- Technique：42.26%［95% CI 34.55, 50.00］／11.25%［95% CI 9.01, 13.82］。
- Timing：71.19%［95% CI 58.93, 82.86］／1.46%［95% CI 0.79, 2.20］。
- Temperature：15.00%［95% CI 0.00, 31.58］／0.32%［95% CI 0.00, 0.75］。

Temperature僅20個正例，區間很寬，不宜單看15%點估計下強結論。

## Baseline證據與限制

完整影片boolean baseline由repo已保存的rows重算，限定相同951步；原來952步中的cc4d-18_101 step09_correct雖target=true但start/end=-1，無可裁切片段，故不納入oracle。主要原標籤accuracy、BA，以及排除Order後的BA，只依GT、原始correctness旗標即可計算，不依賴重建答案的額外假設。

目前無法取得舊Azure VM上完整影片Argon的原始多選答案：主機可達，但現有SSH身分無登入權限。本機及VM2未找到該raw baseline備份。**因此尚無完整影片的去F/G A–E exact-match直接對照**；現有彙整的答對／答錯不能推回具體選項，不能補造。

Baseline的回答true比例、Order-only重標敏感度使用重建boolean，假設非not_visible且答錯的boolean是反向答案；若原baseline含未在rows標出的非法或缺答，此類指標可能有誤。主要原標籤／排除Order的BA不受此假設影響。完整影片與新oracle不是同時期呼叫，不能確認服務端模型修訂完全一致。

## 設計與重現

同舊oracle的951片段與report IDs，標註區間±5秒、平均53.68秒；來源360p、clip_height480、2fps、相同既有ffmpeg編碼設定。單步context僅task、step、五種錯誤定義；Q1保留原correct標籤，Q2去F/G。單步模板移除共用非空錯誤清單範例，不改其他實驗的prompt。

程式位於獨立branch `codex/argon-oracle-single-20261125`，base `4bd16a1`，新程式／結果尚未commit或push；既有Qwen評分腳本未改，僅重用其標籤與解析函式，另建Argon入口。

VM root：`/home/nyu-data/argon_oracle_20261125`；本機worktree：`/Users/jeremy/Documents/ChatGPT/Capstone/argon_oracle`。不要重跑已完成的付費命令。

```sh
# 已執行；--execute才會付費，既有run會拒絕重建
PYTHONPATH=src ~/envs/capstone-qwen8b/bin/python scripts/run_argon_oracle_full.py --execute
# 以下均為離線分析
PYTHONPATH=src ~/envs/capstone-qwen8b/bin/python -m video_report.cli evaluate --run runs/argon_oracle_remaining931 --ground-truth data/cc4d_oracle_single_remaining931/ground_truth.jsonl
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src ~/envs/capstone-qwen8b/bin/python scripts/summarize_argon_oracle_full.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src ~/envs/capstone-qwen8b/bin/python scripts/analyze_argon_oracle.py
```

VM缺matplotlib，因此圖改由本機既有baseline-report-env生成，未安裝或升級套件。測量及推論設定沒有因此變動。

## 交付與驗證

- [全部統計及CI](metrics.csv)、[機器可讀分析](analysis.json)、[逐步配對boolean](paired_bool.json)。
- [逐次用量、格式及費用](per_call.csv)、[執行及tokens摘要](execution.json)、[原始回覆hash](raw_response_sha256.json)。
- [直接逐錄影重抽的獨立驗證](independent_verification.json)：重新以簡單迴圈算主要及排除Order的paired bootstrap，點估計及CI與向量化分析一致；951份raw API回覆hash已在本機逐一驗證。
- [平衡準確率圖](balanced_accuracy.png)、[分組準確率圖](accuracy_by_group.png)，另有同名PDF供匯出。
- `runs/argon_oracle_pilot20/`與`runs/argon_oracle_remaining931/`保存完整snapshots、原始API JSON、可讀payload（影片以SHA256替代，無API key）、逐次回覆及評分；兩份run均已備份本機。
- 預算並行預留、unknown usage停止、key不寫檔，以及bootstrap群聚／非法清單測試均通過。最終完整套件319 passed、1 skipped（VM缺google.genai，Developer API adapter）、1 deselected（live）；見test_results_delivery.txt。

下一個可識別機制的實驗是「相同單步prompt、同樣兩題、相同文字context，但送完整影片」的配對控制組。這是建議，**沒有新增付費呼叫，也沒有啟動200步控制組或GPU注意力實驗**。
