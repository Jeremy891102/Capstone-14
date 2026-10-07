# 單類定位題巢狀 Pilot（草稿）

這批只測定位題的「問題數量」效果，不是逐題模式重跑不同大小。

- 3 支影片，來自 P09／P02／P04；共 96 個唯一問題。
- 每支影片固定原本 time_range，8 ⊂ 16 ⊂ 32；題目、選項順序、答案不改。
- 每個大小 3 種順序，共 9 個 whole-report benchmark。
- `loc_per_field` 收齊 32 題池，供每題只問一次的 baseline；結果可重用於各大小比較。
- 三個 order 用 seeded shuffle + cyclic rotation，再從大集合篩出小集合。
- 每題三次位置不同，至少跨兩個位置區段；不代表三次獨立的模型隨機性實驗。
- 問題證據涵蓋片段前／中／後；實際計數在 summary.json。
- 排除同名動作（含簡單 put down／leave → put 正規化）與證據區間重疊。

狀態：已離線整理與校驗，尚未核對原始影片、標註可辨識性、題目語意互相提示。
`question_review.md` 包含標準答案，只供人工核對，不能送給模型。
影片需要另行下載到 VIDEO_REPORT_VIDEO_ROOT；沒有包含在 Git。

原始來源：data/hd_epic_q40_v1。provenance.json 保存來源檔案 SHA256 與 seed。
來源本身偏向題目密集時段；這批不代表完整 HD-EPIC benchmark。
2 FPS 可能漏掉短動作，正式跑前要預覽實際處理後影片／影格。

完整使用說明：docs/multi_provider_pilot.md。
重新產生：PYTHONPATH=src python scripts/build_localization_pilot.py --out /path/to/new-output。
脚本拒絕覆寫既有目錄，避免破壞人工核對紀錄。

預計呼叫數（尚未執行）：每個 API source 27 次全報告＋96 次逐題＝123 次，未含重試。
正式先跑一份 8 題報告；沒有取得付費執行許可前，不呼叫任何模型 API。
