# cc4d_dense_v1

一個採用 `video_report.v1` 格式的 CaptainCook4D 基準資料集。每份報告對應一支錄影。
報告會詢問整份食譜的整體狀況,以及每個步驟的情形:該步驟是否做對、犯了哪些錯誤、花了多少時間。
每份報告另含兩個誘餌步驟(decoy),用來測試模型是否會產生幻覺。

- **資料來源:** CaptainCook4D 標註檔(`error_annotations.json`、
  `complete_step_annotations.json`)。影片為 GoPro 360p。
- **產生方式:** `scripts/build_cc4d_dense.py`,seed 為 0。
  ```
  python scripts/build_cc4d_dense.py <cc4d annotation dir> data/cc4d_dense_v1 [--recordings 12_51 ...] [--seed 0]
  ```
  使用相同 seed 重新執行腳本,會得到相同的檔案。
- **檔案:** `manifest.json`、`reports.jsonl`(模型看得到的內容)、`ground_truth.jsonl`
  (答案與證據,僅供評估使用)。格式規則見 `docs/data_contract.md`。
- **評分:** 規則已在 `docs/scoring_rules.md` 中決定,但尚未實作。
  在實作完成前,`video-report evaluate` 會拒絕評估此基準。`video-report validate-data`
  已可檢查每個目標。

## 題目格式

### 原始標註結構(轉換的來源)

所有題目都來自 `error_annotations.json` 中每支錄影的一筆紀錄(以下簡稱「該錄影」),
只有食譜名稱另取自 `complete_step_annotations.json`。用到的欄位如下:

| 原始欄位 | 說明 |
| --- | --- |
| `recording_id` | 錄影編號,例如 `12_51` |
| `activity_id` | 食譜編號 |
| `step_annotations[]` | 該錄影的步驟清單,按 `step_id` 排序 |
| `step_annotations[].step_id` | 步驟編號(只存在 ground truth 證據中,不給模型) |
| `step_annotations[].description` | 步驟描述,格式如 `Rinse-Rinse a tomato`,取 `-` 之後的部分 |
| `step_annotations[].start_time` / `end_time` | 步驟在影片中的起訖秒數;`-1` 表示沒有影像 |
| `step_annotations[].errors[].tag` | 錯誤標籤,例如 `Order Error`,去掉 ` Error` 後綴;共 7 種加 `Other` |
| `complete_step_annotations.json[<recording_id>].activity_name` | 食譜名稱 |

下文的「步驟」指 `step_annotations` 的一筆;「完成的步驟」指 `start_time >= 0` 的步驟。

### 報告層級題目(每份報告 5–6 題)

| 欄位 id | 類型 | 問題 | 答案 | 來自哪部分標註 |
| --- | --- | --- | --- | --- |
| `skipped_steps` | multi_choice | 影片中哪些步驟沒有做? | 遺漏的步驟加上 2 個誘餌(誘餌從未被做過) | 遺漏步驟 = `errors[].tag` 含 `Missing Step Error` 的步驟(以 `description` 命名);選項 = 該錄影所有步驟的 `description` + 2 個誘餌 |
| `order_violated` | bool | 是否有任何步驟違反食譜要求的順序? | 任何標有 `Order Error` 的步驟 | 任一步驟的 `errors[].tag` 含 `Order Error` |
| `num_steps_with_errors` | int | 有多少個食譜步驟做錯或被跳過? | 帶有任何錯誤標籤的步驟數 | `errors[]` 非空的步驟數(`Missing Step` 與 `Other` 也算) |
| `first_step` | single_choice | 以下哪個步驟最先被做? | 開始時間最早的步驟 | 完成的步驟中 `start_time` 最小者;選項 = 完成的步驟的 `description` + 2 個誘餌 |
| `total_duration` | seconds | 從第一個步驟開始到最後一個步驟結束,經過了多少秒? | 最後結束時間 − 最早開始時間 | 完成的步驟中 `max(end_time) − min(start_time)` |
| `longest_step` | single_choice | 以下哪個步驟花的時間最長? | 耗時最長的步驟。當前兩名的耗時相差在 20% 以內時省略此題(384 份報告中有 271 份含此題) | 完成的步驟中 `end_time − start_time` 最大者;選項同 `first_step` |

`skipped_steps`、`first_step`、`longest_step` 的選項為食譜步驟加上 2 個誘餌,順序已打亂。
`skipped_steps` 的選項包含所有步驟(含遺漏的);`first_step`、`longest_step` 只含完成的步驟。

### 步驟層級題目(每個步驟 3 題,含誘餌)

| 欄位 id | 類型 | 問題 | 真實步驟的答案 | 誘餌的答案 | 來自哪部分標註 |
| --- | --- | --- | --- | --- | --- |
| `stepNN_correct` | bool | 步驟「…」是否做對了? | 若步驟沒有錯誤標籤則為 `true` | `"not_visible"` | 該步驟的 `errors[]` 是否為空;題目中的步驟名來自 `description` |
| `stepNN_error_types` | multi_choice | 步驟「…」犯了哪些錯誤? | 錯誤標籤,無錯誤則為 `[]` | `"not_visible"` | 該步驟 `errors[].tag` 對應到 A–G 選項 |
| `stepNN_duration` | seconds | 步驟「…」花了多少秒? | 結束 − 開始 | `"not_visible"` | 該步驟的 `end_time − start_time` |
| (誘餌) | | | | | 誘餌的步驟文字取自**其他食譜**的步驟 `description`;`step_id` 與來源食譜名稱只記在 ground truth 證據中 |

`error_types` 的選項固定為以下 7 個:A Preparation(準備)、B Measurement(計量)、
C Technique(技巧)、D Timing(時間)、E Temperature(溫度)、F Order(順序)、
G Missing Step(遺漏步驟)。

有兩種情況會減少步驟層級的題數:

- **遺漏步驟**(`start_time = -1`,沒有影像)沒有 `duration` 題。其 `correct` 答案為 `false`,
  `error_types` 答案為 `["G"]`(Missing Step)。
- 標為 **`Other`** 的步驟(資料集中共 8 個)沒有 `error_types` 題,因為 `Other`
  不在選項之內。

每個真實步驟的 ground truth 證據都附上原始的 `step_id`、`start_time`、`end_time`、`errors`,方便回頭對照原始標註。

### 其他欄位的來源

| 報告欄位 | 來自哪部分標註 |
| --- | --- |
| 報告 id `cc4d-<recording_id>` | `error_annotations.json` 的 `recording_id` |
| `input.context.task_name` | `complete_step_annotations.json[<recording_id>].activity_name`(找不到時退回 `activity_id`) |
| `time_range.end` | 完成的步驟的 `max(end_time) + 5` 秒 |
| 影片 `<recording_id>_360p.mp4` | 以 `recording_id` 命名的 GoPro 360p 影片(非標註檔) |

### 答案格式

| 類型 | 答案格式 |
| --- | --- |
| `single_choice` | 一個標籤:`"A"` … `"Z"`,超過 26 個選項後為 `"AA"`、`"AB"`、… |
| `multi_choice` | 一串不重複的標籤,例如 `["C", "A"]`;`[]` 表示沒有 |
| `bool` | `true` / `false` |
| `int` | 整數 |
| `seconds` | 大於等於 0 的數字,四捨五入到 0.1 秒 |
| 任何類型 | `"not_visible"`。每個欄位都允許此值(`allow_not_visible: true`),因此陷阱欄位不會顯得特別 |

### 防止洩漏答案的設計

- **步驟以描述命名,絕不使用編號。** CaptainCook4D 的 `step_id` 順序並不是食譜應該執行的順序。
- **重複的描述會加上序數。** 當食譜中某個步驟重複出現(例如攪拌兩次),各份副本會依實際執行的先後標示為「(1st time)」、「(2nd time)」。
- **步驟區塊會被打亂,並在打亂後重新編號為 `step01`…`stepNN`。** 這樣欄位順序不會暴露真實的步驟順序,欄位 id 也不會暴露哪些是誘餌。原始的 `step_id` 只保留在 ground truth 的證據中。
- **每份報告有兩個來自其他食譜的誘餌:**
  - **明顯**的誘餌,與本食譜沒有任何共同的實詞,例如在 Microwave Egg Sandwich 中出現「Add chopped cilantro to the bowl」;
  - **擬真**的誘餌,從與本食譜用詞重疊最多的 10 個步驟中隨機挑選,例如「Add 1/3 cup cheddar cheese to a microwave-safe cup」。

  候選誘餌若與本食譜任一步驟的詞彙 Jaccard 相似度 ≥ 0.6(`NEAR_DUP`),會先被排除,避免誘餌其實是本食譜某步驟的改寫(例如只差冠詞或標點)。

  誘餌也會出現在每一個步驟選項清單中。

## 模型的輸入

- **影片:** `<recording_id>_360p.mp4`,從 0 秒傳送到最後一個已標註步驟之後 5 秒(`time_range`)。
- **背景資訊:** `task_name`,即食譜名稱。只有在實驗白名單允許時才會傳送。
- **不另外提供標準步驟清單。** 食譜的步驟只會以步驟題選項的形式出現。

## 資料分布

### 影片

| | 數值 |
| --- | --- |
| 報告數(錄影數) | 384 |
| 食譜數 | 24(每個食譜 12–20 支錄影) |
| 片長(分鐘) | 中位數 13.1,平均 14.5,範圍 3.3–41.1 |
| 傳送的影片總長 | 93.1 小時 |
| 每份報告的食譜步驟數 | 中位數 14,平均 14.8,範圍 7–26 |
| 至少有一個錯誤的報告 | 220(57%) |
| 每份報告的錯誤數 | 中位數 2,平均 5.1,最大 23 |
| 有順序違規的報告 | 117 |
| 有被跳過步驟的報告 | 140(共 285 個被跳過的步驟) |

片長為 `time_range.end`,即最後一個已標註步驟的結束時間加 5 秒。完整影片檔可能稍長。

### 每份報告的題數

| | 數值 |
| --- | --- |
| 總題數 | 21,300 |
| 每份報告 | 中位數 54,平均 55.5,範圍 30–87 |
| 公式 | 3 ×(步驟數 + 2 個誘餌)+ 5 或 6,再減去每個遺漏步驟 1 題與每個 `Other` 步驟 1 題 |

| 每份報告題數 | 報告數 |
| --- | --- |
| 30–39 | 34 |
| 40–49 | 120 |
| 50–59 | 89 |
| 60–69 | 96 |
| 70–79 | 20 |
| 80–87 | 25 |

### 題目類型

| 類型 | 數量 | 占比 |
| --- | --- | --- |
| bool | 6,852 | 32.2% |
| multi_choice | 6,844 | 32.1% |
| seconds | 6,565 | 30.8% |
| single_choice | 655 | 3.1% |
| int | 384 | 1.8% |

| 欄位 | 數量 |
| --- | --- |
| `stepNN_correct` | 6,468(5,700 真實 + 768 誘餌) |
| `stepNN_error_types` | 6,460 |
| `stepNN_duration` | 6,181(5,413 真實 + 768 誘餌) |
| `skipped_steps`、`order_violated`、`num_steps_with_errors`、`first_step`、`total_duration` | 各 384 |
| `longest_step` | 271 |

### 答案

| | 數值 |
| --- | --- |
| `"not_visible"` 目標(誘餌題) | 2,304(10.8%,每份報告 6 題) |
| 做對的真實步驟(`correct = true`) | 5,700 中的 3,736(65.5%) |
| `error_types` 答案依標籤數分布 | 0 個標籤:3,736;1 個:1,437;2 個:454;3 個:64;4 個:1 |
| 錯誤標籤頻率 | Order 791、Technique 498、Preparation 394、Measurement 331、Missing Step 285、Timing 177、Temperature 65 |
| 步驟耗時(秒) | 中位數 35.0,十分位數 9.8 / 15.7 / 21.6 / 28.1 / 35.0 / 44.1 / 57.6 / 76.9 / 118.3,範圍 1.0–1068.2 |
| 選項數(步驟清單) | 7–28;只有 `cc4d-23_5 first_step` 的答案是兩個字母(`AA`) |

## 已知限制

- **對每個步驟都回答「正確、無錯誤」是很強的基準線。** 對真實步驟而言,`correct` 可達 65.5%,
  `error_types` 可達 65.6%(5,692 中的 3,736)。回報準確率時請與此基準線比較。
- **`Order` 是最常見的錯誤標籤,** 但判斷它需要整支影片的順序資訊,而不是單一步驟。
- **明顯的誘餌光看食譜名稱就能辨認。** 擬真的誘餌才是真正的考驗。擬真程度來自詞彙重疊的啟發式方法,尚未經人工檢查。
- **非常短的步驟是以標註雜訊來評分。** 耗時在 max(2 秒, 目標值的 25%) 之內視為正確
  (見 `docs/scoring_rules.md`)。對於 395 個短於 8 秒的真實步驟,這相當於 2 秒的下限,
  與標註邊界雜訊的大小相近。
- **腳本已處理的標註怪異之處:**
  - 53 支錄影將同一個 `step_id` 用於兩個不同的描述。
  - 部分步驟的時間戳為 −1,卻沒有 Missing Step 標籤;這些步驟不會產生耗時題。
