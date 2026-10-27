# CC4D 題型實驗:出題規則(草稿,2026-10-26)

說明如何從 CaptainCook4D 原始標注產生各題型的題目。實驗設計見
`docs/cc4d_answer_type_proposal.md`,評分規則見 `docs/scoring_rules.md`。

## 原則

- **先抽事實,題型只是問法。** 同一個事實在不同 call 裡用各種題型各問一次;同一個 call 裡每個
  步驟只出現一次。
- **題型分配不看答案,選項清單不隨答案改變。**
- 題目用英文;步驟名稱沿用 `cc4d_dense_v1`(描述文字、重複步驟加序數、不給 `step_id`)。

## 模型輸入

- **影片:** 整段錄影,0 s 到最後一個標注步驟後 5 s,fps = 2。
- **context:** `task_name`、`recipe`、`error_definitions`。

## 使用的標注

| 欄位 | 用途 |
| --- | --- |
| `start_time` / `end_time` | 步驟何時做、做多久;`-1` 表示沒有畫面 |
| `errors[].tag` | 錯誤類別;`Missing Step` 表示沒做 |
| `errors[].description` | open_text 的參考答案 |
| `modified_description` | 只給評審參考 |
| task graph `edges` | 找會被 recipe 洩漏的題目;節點 id 和 `step_id` 不同,用描述文字對應 |

## 矩陣

| 內容 \ 題型 | bool | single | multi | seconds | int(計數) | open_text |
| --- | --- | --- | --- | --- | --- | --- |
| 錯誤(4,696 步) | ✓ | 6 選項 | 6 選項 | — | ✓ | ✓ |
| 時長(3,763 步) | ✓ | 4 區間 | 8 步 | ✓ | 8 步 | — |
| 定位(5,049 步) | ✓ | 5 區間 | 8 步 | ✓ | 8 步 | — |

- 步驟數是 `cc4d_dense_v1` 全部 350 份報告的題庫;實驗用的 64 份(主實驗)和 E5 的 125 / 44 份
  見實驗提案。
- 空格:標注沒有錯誤發生的時間點;時長和定位的開放題不易評分。
- int 一律是計數,不問秒數。
- 「8 步」是一題列出 8 個步驟,每份報告每列只能出一題。

## 錯誤

只用 5 類:Preparation、Measurement、Technique、Timing、Temperature。Order 要看整支影片的
順序,歸到補充內容;Missing Step 改作幻覺題。

| 步驟(`cc4d_dense_v1` 5,295 步) | 數量 | 處理 |
| --- | --- | --- |
| 有畫面,沒有錯誤 | 3,591 | 出題 |
| 有畫面,有 5 類之一 | 1,118 | 出題;Order 忽略 |
| 只有 Order 錯誤 | 334 | 排除(不能當作沒有錯誤) |
| Other | 6 | 排除 |
| 參考描述太模糊(清單見 `scoring_rules.md`) | 13 | 排除 |
| Missing Step,沒有畫面 | 237 | 幻覺題,答案 `null` |
| Missing Step 但有畫面 | 3 | 排除 |
| 沒有畫面、沒有 Missing Step | 6 | 排除 |

共 4,696 步。標籤數:0 個 3,591、1 個 962、2 個 137、3 個 6。一律答沒有錯誤是 76.5%,所以
實驗抽 4 錯 4 對,並分開報有錯 / 沒錯兩組。

| 題型 | 問法 | 答案 |
| --- | --- | --- |
| bool | Was the step "X" done correctly? | 沒有 5 類標籤 |
| single | Which of these mistakes was made in the step "X"? If several were made, pick any one. 5 類 + No mistake | 任一個真錯誤都算對 |
| multi | Which of these mistakes were made in the step "X"? 同樣 6 個選項 | 標籤集合;No mistake 只能單選 |
| int | How many different kinds of mistakes were made in the step "X"? | 0–3 |
| open_text | Compared with the recipe, what did the person do differently in the step "X"? Answer "none" if the step was done as written. | 錯誤描述,或 none |

## 時長

題庫:有畫面的 5,049 步,排除描述寫了時間的步驟(看文字就有四成機會答對),剩 4,527;再排除
時長在切點 17.1 / 31.3 / 56.5 s ±2 s 內的步驟,剩 3,763 步。

| 題型 | 問法 | 答案 |
| --- | --- | --- |
| bool | Did the step "X" take longer than 31 seconds? | > 31.3 s,約一半是 |
| single | How long did the step "X" take? 4 個區間 | 所在區間,各區間數量相近 |
| multi | Which of these steps took longer than 31 seconds? 8 步 | > 31.3 s 的步驟 |
| seconds | How many seconds did the step "X" take? | `end − start` |
| int | Among these steps, how many took longer than 31 seconds? 同樣 8 步 | 步驟數 |

## 定位

題庫:有畫面的 5,049 步。

| 題型 | 問法 | 答案 |
| --- | --- | --- |
| bool | Was the step "X" being done between t1 and t2? | 是否重疊,一半重疊 |
| single | When was the step "X" done? 5 個區間 | 正確區間;另 4 個取其他步驟、不重疊 |
| multi | Which of these steps were being done between t1 and t2? 8 步 | 重疊的步驟 |
| seconds | At what time did the step "X" start? | `start_time`,誤差 `max(2, 0.25 × 步驟時長)` |
| int | Among these steps, how many were being done between t1 and t2? 同樣 8 步 | 步驟數 |

干擾選項盡量挑 task graph 中和 X 沒有依賴關係的步驟,避免 recipe 洩漏時間位置。

## 沒做的步驟(幻覺)

- **來源:** Missing Step 且沒有畫面的步驟,題庫共 237 個;主實驗的 64 份報告中有 108 個(47 份)。
  它們在 recipe 裡真實存在,只能靠影片判斷。
- **一題一步:** 每題附 "If the step was not done in the video, answer null."。沒做的步驟答其他
  內容是幻覺;有做的步驟答 `null` 是漏看。
- **一題 8 步:** 沒做的步驟可當選項,選了或算進去就是幻覺。
- 只在 E1 問。錯誤列另外看誤報(沒錯的步驟說成有錯)、`extras`、`extra_claims`。

## 補充內容(不進主矩陣)

- **順序:** 「X 比 Y 先做嗎?」只挑沒有依賴、時間不重疊的步驟對。每份報告中位數 13 對,彼此
  共用步驟,不是獨立事實。
- **遺漏:** 沿用 `skipped_steps`。

## 擱置

- 時長和定位列的 bool / multi / int 問的是「超過 31.3 s」「在 [t1, t2] 內」這類判斷,和 single /
  seconds 不是同一個事實。理想是像錯誤列一樣每步都能用每種題型問同一個事實。
