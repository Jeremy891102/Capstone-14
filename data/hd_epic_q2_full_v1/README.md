# hd_epic_q2_full_v1 概覽

Q2（子類型準確率）擴充題集：hd_epic_q40_v1 那 76 支影片上，所有 how / why / weight 單選題。不需要新影片。

- 報告數：276（69 個沿用 q40_v1 窗口 + 207 個新窗口）
- 總題數：865（how 421、why 405、weight 39），影片數：how 67、why 72、weight 25
- Localization 不在這裡，沿用 E1 在 q40_v1 上的 2861 題結果。

## 取樣方式

1. 已在 q40_v1 的 351 題（`hdepic-dense-*`）：窗口、題目文字、ground truth 原封不動，E1 的預測可直接沿用。
2. 其餘題目（`hdepic-q2-*`）：篩選條件同 `build_hd_epic_dense.py`（單一影片、有 start/end、選項不重複）。how 有 1 題因選項重複被排除（500 題中 422 題在這 76 支影片上）。
3. 新窗口長度 = 同一支影片在 q40_v1 的窗口長度，讓每題送進模型的影片長度與 E1 相同分布。
4. 窗口位置：對最早未覆蓋的題目，起點在「題目片段完整落在窗口內」的範圍內隨機取（seed 20261022），再限制在影片長度內；窗口內其他未覆蓋題目一併收進，重複直到全部覆蓋。
5. `video_durations.json` 是在 Azure VM 上用 ffmpeg 量的影片長度。
6. 產生方式：`python scripts/build_hd_epic_q2_full.py <vqa 目錄> data/hd_epic_q40_v1 data/hd_epic_q2_full_v1 --durations data/hd_epic_q2_full_v1/video_durations.json`

跑的時候用 `per_field`（一題一次 call），只需要跑 `hdepic-q2-*` 的 514 題。
