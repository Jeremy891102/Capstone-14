# cc4d_dense_v1

一個採用 `video_report.v1` 格式的 CaptainCook4D 基準資料集。每份報告對應一支錄影。
報告會詢問整份食譜的整體狀況,以及每個步驟的情形:該步驟是否做對、犯了哪些錯誤、花了多少時間。

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

### 報告層級題目(每份報告 4–5 題)

| 欄位 id | 類型 | 問題 | 答案 | 來自哪部分標註 |
| --- | --- | --- | --- | --- |
| `skipped_steps` | multi_choice | 影片中哪些步驟沒有做? | 遺漏的步驟(可以為空) | 遺漏步驟 = `errors[].tag` 含 `Missing Step` 的步驟(以 `description` 命名);選項 = 全部遺漏步驟 + 從其餘步驟隨機補足的真實步驟,共 8 個 |
| `order_violated` | bool | 是否有任何步驟違反食譜要求的順序? | 任何標有 `Order Error` 的步驟 | 任一步驟的 `errors[].tag` 含 `Order Error` |
| `steps_done` | int | 影片中實際做了多少個食譜步驟? | 有影像的步驟數(最小 8,中位數 14,最大 25) | `start_time >= 0` 的步驟數;遺漏步驟(`start_time = -1`)與時間戳為 −1 但沒有 Missing Step 標籤的步驟都不算 |
| `total_duration` | seconds | 從第一個步驟開始到最後一個步驟結束,經過了多少秒? | 最後結束時間 − 最早開始時間 | 完成的步驟中 `max(end_time) − min(start_time)` |
| `longest_step` | single_choice | 以下哪個步驟花的時間最長? | 耗時最長的步驟。當前兩名的耗時相差在 20% 以內時省略此題(350 份報告中有 255 份含此題) | 完成的步驟中 `end_time − start_time` 最大者;選項 = 答案 + 從其餘完成的步驟隨機挑 7 個,共 8 個 |

`skipped_steps`、`longest_step` 的選項**固定為 8 個**真實步驟,順序已打亂。
這樣食譜的步驟數不會影響選項數,也就不會成為影響模型表現的潛在變因。

- `skipped_steps`:8 個選項包含**全部**遺漏步驟,不足的部分從其餘步驟隨機補足(可能包含完成的步驟)。
  資料集中遺漏步驟最多 6 個,所以 8 個名額一定放得下。
- `longest_step`:8 個選項只取有影像的步驟,其中 1 個是答案,其餘 7 個隨機挑選。
- 有影像的步驟少於 8 個的錄影(共 19 份)無法湊滿選項,**整份報告已從資料集中移除**。
- 另有 15 份報告被移除:這些錄影的步驟名稱(例如重複步驟的「(1st time)」,或與 task graph 寫法不同的步驟)在 `recipe` 食譜文字中找不到對應,題目與食譜清單會對不上。
- 因此報告數從 384 降為 350。

### 步驟層級題目

食譜中的每個步驟,固定問下表的 3 題(遺漏步驟與 `Other` 步驟的例外見下文)。

| 欄位 id | 類型 | 問題 | 答案 | 來自哪部分標註 |
| --- | --- | --- | --- | --- |
| `stepNN_correct` | bool | 步驟「…」是否做對了? | 若步驟沒有錯誤標籤則為 `true` | 該步驟的 `errors[]` 是否為空;題目中的步驟名來自 `description` |
| `stepNN_error_types` | multi_choice | 步驟「…」犯了哪些錯誤? | 錯誤標籤,無錯誤則為 `[]` | 該步驟 `errors[].tag` 對應到 A–G 選項 |
| `stepNN_duration` | seconds | 步驟「…」花了多少秒? | 結束 − 開始 | 該步驟的 `end_time − start_time` |

`error_types` 的選項固定為以下 7 個:A Preparation(準備)、B Measurement(計量)、
C Technique(技巧)、D Timing(時間)、E Temperature(溫度)、F Order(順序)、
G Missing Step(遺漏步驟)。
定義見 `input.context.error_definitions`;論文中沒有 `Other` 類別的定義,所以 `Other` 步驟不出 `error_types` 題。

有兩種情況會減少步驟層級的題數:

- **遺漏步驟**(`start_time = -1`,沒有影像)沒有 `duration` 題。其 `correct` 答案為 `false`,
  `error_types` 答案為 `["G"]`(Missing Step)。
- 標為 **`Other`** 的步驟(本資料集中共 6 個)沒有 `error_types` 題,因為 `Other`
  不在選項之內。

每個真實步驟的 ground truth 證據都附上原始的 `step_id`、`start_time`、`end_time`、`errors`,方便回頭對照原始標註。

### 其他欄位的來源

| 報告欄位 | 來自哪部分標註 |
| --- | --- |
| 報告 id `cc4d-<recording_id>` | `error_annotations.json` 的 `recording_id` |
| `input.context.task_name` | `complete_step_annotations.json[<recording_id>].activity_name`(找不到時退回 `activity_id`) |
| `input.context.recipe` | 官方 annotations 倉庫 `task_graphs/<食譜>.json`(步驟描述與 `edges` 前置關係),不在 `error_annotations.json` 中 |
| `input.context.error_definitions` | CaptainCook4D 論文(arXiv 2312.14556)第 3.2 節的錯誤類別定義,不在標註檔中 |
| `time_range.end` | 完成的步驟的 `max(end_time) + 5` 秒 |
| 影片 `<recording_id>_360p.mp4` | 以 `recording_id` 命名的 GoPro 360p 影片(非標註檔) |

### 答案格式

| 類型 | 答案格式 |
| --- | --- |
| `single_choice` | 一個標籤:`"A"` … `"H"`(固定 8 個選項);格式上支援超過 26 個選項後的 `"AA"`、`"AB"`、… |
| `multi_choice` | 一串不重複的標籤,例如 `["C", "A"]`;`[]` 表示沒有 |
| `bool` | `true` / `false` |
| `int` | 整數 |
| `seconds` | 大於等於 0 的數字,四捨五入到 0.1 秒 |

### 防止洩漏答案的設計

- **步驟以描述命名,絕不使用編號。** CaptainCook4D 的 `step_id` 順序並不是食譜應該執行的順序。
- **重複的描述會加上序數。** 當食譜中某個步驟重複出現(例如攪拌兩次),各份副本會依實際執行的先後標示為「(1st time)」、「(2nd time)」。
- **步驟區塊會被打亂,並在打亂後重新編號為 `step01`…`stepNN`。** 這樣欄位順序不會暴露真實的步驟順序。原始的 `step_id` 只保留在 ground truth 的證據中。

> **關於幻覺測試:** 早期版本每份報告含 2 個來自其他食譜的誘餌步驟(答案為 `"not_visible"`)。
> 資料集預設會提供食譜(`recipe`),誘餌不在食譜清單中,一眼就能認出,已失去測試意義,**現已取消**。
> 「模型是否把沒做的步驟說成做了」改由 `skipped_steps` 與 `Missing Step` 測試。

## 模型的輸入

- **影片:** `<recording_id>_360p.mp4`,從 0 秒傳送到最後一個已標註步驟之後 5 秒(`time_range`)。
- **背景資訊(`input.context`):** 資料集預設提供三個欄位,實驗必須在 `prompt.context_keys`
  中列出才會真的給模型,例如 `context_keys: [task_name, error_definitions, recipe]`:
  - `task_name`:食譜名稱。
  - `error_definitions`:`error_types` 七個類別(Preparation、Measurement、Technique、Timing、
    Temperature、Order、Missing Step)的定義,英文,每個類別一行。沒有這個欄位時,模型只看得到類別名稱。
  - `recipe`:該食譜的步驟清單與每個步驟的前置步驟。內容取自 CaptainCook4D 官方 annotations
    倉庫的 task graph。每一行是一個步驟的描述,後面以 `(after: …)` 列出必須先完成的步驟。
    沒有 `after` 的步驟可在任何時間做,互不依賴的步驟可以任意先後。
    清單按原始 `step_id` 排序,不帶步驟編號,所以清單順序本身不是正確順序。
- **正確順序是偏序,不是唯一序列。** 同一食譜的無錯誤錄影順序各不相同,因為許多步驟互不依賴。
- **沒有「每個步驟的正確做法」。** 食譜只有每步一句描述,沒有更詳細的說明。
  Measurement、Timing、Temperature 的規格通常寫在描述裡;Technique、Preparation 沒有明確標準。

## 資料分布

### 影片

| | 數值 |
| --- | --- |
| 報告數(錄影數) | 350(原 384 份中,移除 19 份有影像步驟少於 8 個者,再移除 15 份步驟名稱與食譜文字對不上者) |
| 食譜數 | 23(每個食譜 11–18 支錄影) |
| 片長(分鐘) | 中位數 13.6,平均 15.0,範圍 3.9–41.1 |
| 傳送的影片總長 | 87.2 小時 |
| 每份報告的食譜步驟數 | 中位數 15,平均 15.1,範圍 9–25 |
| 至少有一個錯誤的報告 | 192(55%) |
| 每份報告的錯誤數 | 中位數 1,平均 4.9,最大 23 |
| 有順序違規的報告 | 100 |
| 有被跳過步驟的報告 | 121(共 240 個被跳過的步驟) |

片長為 `time_range.end`,即最後一個已標註步驟的結束時間加 5 秒。完整影片檔可能稍長。

### 每份報告的題數

| | 數值 |
| --- | --- |
| 總題數 | 17,291 |
| 每份報告 | 中位數 47,平均 49.4,範圍 31–80 |
| 公式 | 3 ×(步驟數)+ 4 或 5,再減去每個遺漏步驟 1 題與每個 `Other` 步驟 1 題 |

| 每份報告題數 | 報告數 |
| --- | --- |
| 30–39 | 111 |
| 40–49 | 74 |
| 50–59 | 86 |
| 60–69 | 49 |
| 70–79 | 22 |
| 80 | 8 |

### 題目類型

| 類型 | 數量 | 占比 |
| --- | --- | --- |
| bool | 5,645 | 32.6% |
| multi_choice | 5,639 | 32.6% |
| seconds | 5,402 | 31.2% |
| single_choice | 255 | 1.5% |
| int | 350 | 2.0% |

| 欄位 | 數量 |
| --- | --- |
| `stepNN_correct` | 5,295 |
| `stepNN_error_types` | 5,289 |
| `stepNN_duration` | 5,052 |
| `skipped_steps`、`order_violated`、`steps_done`、`total_duration` | 各 350 |
| `longest_step` | 255 |

### 答案

| | 數值 |
| --- | --- |
| 做對的真實步驟(`correct = true`) | 5,295 中的 3,597(67.9%) |
| `error_types` 答案依標籤數分布 | 0 個標籤:3,597;1 個:1,244;2 個:390;3 個:57;4 個:1 |
| 錯誤標籤頻率 | Order 690、Technique 441、Preparation 339、Measurement 279、Missing Step 240、Timing 153、Temperature 57 |
| 步驟耗時(秒) | 中位數 34.85,十分位數 9.9 / 15.8 / 21.5 / 28.0 / 34.9 / 44.0 / 57.3 / 77.2 / 119.5,範圍 1.0–1068.2 |
| 選項數(步驟清單) | 固定 8 個,答案標籤皆為單一字母 A–H |

## 已知限制

- **對每個步驟都回答「正確、無錯誤」是很強的基準線。** 對真實步驟而言,`correct` 可達 67.9%,
  `error_types` 可達 68.0%(5,289 中的 3,597)。回報準確率時請與此基準線比較。
- **`Order` 是最常見的錯誤標籤,** 但判斷它需要整支影片的順序資訊,而不是單一步驟。
- **沒有幻覺測試題。** 資料集不含誘餌步驟,只能用 `skipped_steps` 與 `Missing Step` 間接檢查模型是否把沒做的步驟說成做了。
- **非常短的步驟是以標註雜訊來評分。** 耗時在 max(2 秒, 目標值的 25%) 之內視為正確
  (見 `docs/scoring_rules.md`)。對於 358 個短於 8 秒的真實步驟,這相當於 2 秒的下限,
  與標註邊界雜訊的大小相近。
- **腳本已處理的標註怪異之處:**
  - 38 支錄影有重複的步驟描述(不同 `step_id` 但文字相同),以「(1st time)」等序數區分。
  - 1 支錄影將同一個 `step_id` 用於兩個不同的描述。
  - 部分步驟的時間戳為 −1,卻沒有 Missing Step 標籤;這些步驟不會產生耗時題。
