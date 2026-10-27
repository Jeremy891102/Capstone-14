# 各題型的評分規則

英文原版是 `docs/scoring_rules.md`,兩份內容相同,有出入時以英文版為準。

每種題型怎麼判對錯,從解析出的答案到最後的指標。欄位和答案的格式定義在
`docs/data_contract.md`。

**狀態(2026-10-13):**

- 只有 `single_choice` 已實作,在 `evaluation/scorers.py`。其他題型的規則在這裡定好了,但還沒有
  程式碼。在實作之前,`video-report evaluate` 會拒絕含有這些題型的 run。
- 以下規則是設計 CaptainCook4D benchmark(`cc4d_dense_v1`)時定的,適用於任何使用這些題型的
  benchmark。
- `cc4d_dense_v1` 使用 `bool`、`int`、`seconds`、`single_choice` 和 `multi_choice`。
- 2026-10-26 為 CC4D 題型實驗新增(`docs/cc4d_format_questions.md`):`open_text`、有多個可接受
  答案的 `single_choice`、`seconds` 的時間點題,以及跨題型比較(指標 6)。

## 原則

1. **每個欄位只有一個主要結果:對或錯。** 整體準確率是所有題型的平均。分母是所有選中的欄位,
   missing、invalid 和沒執行的答案都算錯,和現在相同。
2. **有些題型另外記錄次要測量**(誤差、選項計數)。只用於分析,不會改變主要結果。
3. **答案嚴格依 JSON 型別檢查。** 型別或形狀不對就是 `invalid_answer`,不做轉換。格式錯誤和答錯
   分開記錄。例如:bool 題回答字串 `"true"`、seconds 題回答 `"30 s"`。

## 結果類別

以下六種結果適用於每種題型,和現有的 scorer 相同:

| 結果 | 意思 |
| --- | --- |
| `correct` | 有效答案,且符合該題型的規則 |
| `incorrect` | 有效答案,但不符合該題型的規則 |
| `invalid_answer` | 有這個 key,但值不是這個欄位題型的有效答案 |
| `missing_answer` | 回應可以解析,但沒有這個欄位的 key |
| `invalid_output` | 整個回應無法使用(空的、不是合法 JSON、key 重複、不是物件) |
| `not_executed` | 沒有任何成功的 provider call 涵蓋這個欄位 |

## 沒做的步驟(`null`)

2026-10-26 為 CC4D 題型實驗決定。問單一步驟的欄位可以允許回答 `null`,意思是「影片裡沒有做這一
步」。每種題型的題目都用同一句話說明這一點。

- **有效答案:** 欄位允許時,`null` 在每種題型都是有效答案。
- **正確:** 只有答案是 `null` 時(Missing Step、沒有畫面),回答 `null` 才算對。這類欄位回答其他
  任何內容都算錯,記為**幻覺**。
- 有畫面的步驟回答 `null` 算錯,記為**漏看**。
- 兩種比例都要在準確率旁邊單獨報告。
- 一題列出多個步驟當選項的欄位(CC4D 時長和定位的 `multi_choice` / `int`)不使用 `null`:沒做的
  步驟可以是選項之一,選了或算進去就是幻覺。

## 各題型規則

### `single_choice`(已實作)

- **有效答案:** 該欄位選項中的一個標籤:`"A"` … `"Z"`,接著是 `"AA"`、`"AB"`……
- **正確:** 標籤等於答案。
- **多個可接受答案**(2026-10-26 決定):答案可以是一組標籤,答中其中任一個就算對。用於 CC4D 的
  錯誤類別題:有兩個錯誤的步驟,答哪一個都對。題目文字必須寫明 "pick any one"。每一題的選項都
  相同,所以題型不會因答案而改變。

### `bool`

- **有效答案:** JSON 的 `true` 或 `false`。
- **正確:** 等於答案。

### `int`

- **有效答案:** JSON 整數。`true` / `false` 和 `3.0` 都無效。
- **正確:** 完全等於答案。沒有容許誤差,連 ±1 都不行(2026-10-08 決定)。
- **另外記錄:** `abs_error = |回答 − 答案|`。

### `seconds`

- **有效答案:** 有限的 JSON 數字,≥ 0,整數或小數都可以。
- **正確:** `|回答 − 答案| ≤ max(2, 0.25 × 答案)`。容許誤差以**答案**計算,不是以回答計算。
  - 8 秒以下用 2 秒下限,大約涵蓋標注邊界的雜訊。
  - 8 秒以上用 25%。

  | 答案 | 接受範圍 |
  | --- | --- |
  | 4 s | 2–6 s |
  | 8 s | 6–10 s |
  | 40 s | 30–50 s |
  | 120 s | 90–150 s |

- **另外記錄:** `abs_error = |回答 − 答案|`,以及 `rel_error = abs_error / 答案`(答案為 0 時是
  null)。
- **可設定:** 兩個容許值(下限 2 秒、比例 0.25)放在評分設定裡,每次評分都記錄用了哪些值。改了
  只要重新評分,不用重跑。
- **時間點**(2026-10-26 決定,例如「步驟 X 在什麼時候開始?」):上面的容許誤差會隨時間點變大
  (600 秒時是 ±150 秒),所以時間點欄位改用步驟本身的長度計算容許誤差:
  `|回答 − start| ≤ max(2, 0.25 × (end − start))` 就算對。下限和比例與時長題相同。這種欄位會標記
  自己是時間點,並在答案資料中帶有步驟長度。

### `multi_choice`

- **有效答案:** 一個 JSON 陣列,裡面是互不重複、且都在該欄位選項中的標籤。順序不重要。`[]` 表示
  「都沒有」。
  - 有重複或未知標籤的陣列無效。
- **正確:** 回答的標籤集合和答案的標籤集合完全相同。答案是 `[]` 時,只有 `[]` 算對。
- **互斥選項**(2026-10-26 決定):欄位可以把某個選項標為互斥,例如 CC4D 的 "No mistake",讓
  選項清單和同一題的 `single_choice` 版本相同。這個選項只能單獨選:`["F"]` 有效,`["F", "A"]` 是
  `invalid_answer`。此時「沒有」的答案是 `["F"]`,`[]` 是 `invalid_answer`。計算 hits / misses /
  extras 時,互斥選項和其他選項一樣計算。
- **另外記錄**(每個有效答案):

  | 計數 | 意思 | 例子:答案 `["A", "C"]`,回答 `["C", "D"]` |
  | --- | --- | --- |
  | `hits` | 答案和回答都有的標籤 | 1(C) |
  | `misses` | 答案有、但沒回答的標籤 | 1(A) |
  | `extras` | 回答了、但答案沒有的標籤 | 1(D) |

  之後可以推出 precision 和 recall:`hits / (hits + extras)` 和 `hits / (hits + misses)`。

### `open_text`

自由文字的回答,由 LLM 對照一段簡短的參考答案評分。2026-10-26 為 CC4D 逐步驟題目決定:
"Compared with the recipe, what did the person do differently in the step "…"? Answer "none" if
the step was done as written."

- **有效答案:** 非空的 JSON 字串。
- **評審的輸入**(只有文字,沒有影片):步驟描述、參考答案(一組錯誤描述,或 `none`)、該步驟的
  `modified_description`(有的話),以及回答。
- **正確:**
  1. 參考答案是 `none`:回答說沒有做得不同才算對。提出任何錯誤都算錯。
  2. 參考答案有錯誤:回答講出**至少一個**參考錯誤實際做錯了什麼,才算對。只講出錯誤類別,或像
     「做得不好」這種籠統說法,不算。用詞不同但意思相同,算對。
  3. 多講的錯誤不影響結果。
- **評審的輸出:** `{"correct": bool, "matched": [參考答案的編號], "extra_claims": int}`。
- **另外記錄:** `matched` 和 `extra_claims`(對應 `multi_choice` 的 `extras`)。
- **可設定:** 評審模型、評審 prompt 版本,以及比對規則(「至少一個」或「全部」)放在評分設定裡,
  每次評分都記錄。改了只要重新評分,不用重跑。
- **驗證:** 正式使用前,人工標約 100 個回答,報告和評審的一致性。

**CC4D 題庫**(`cc4d_dense_v1` 的 5,295 個步驟):

| 步驟 | 數量 | 處理 |
| --- | --- | --- |
| 有畫面,沒有錯誤 | 3,591 | 出題,參考答案 `none` |
| 有畫面,至少有一個 Technique / Preparation / Measurement / Timing / Temperature 錯誤 | 1,118 | 出題,參考答案只用這幾類錯誤的描述(Order 的描述不用) |
| 有畫面,只有 Order 錯誤 | 334 | 不出(順序要看整支影片,不是看這一步) |
| Other | 6 | 不出 |
| Missing Step,沒有畫面 | 237 | 當幻覺題,答案 `null`(見上方「沒做的步驟」) |
| Missing Step 但有畫面 | 3 | 不出(標注矛盾) |
| 沒有畫面、沒有 Missing Step 標籤 | 6 | 不出(選做步驟 "Top with more parmesan if desired") |

所以有畫面的步驟 4,709 個,扣掉下面 13 個手動排除:4,696 個。一律回答 `none` 可以得到 76.5%
(3,591 / 4,696)。錯誤列的每種題型都用這同一組 4,696 個步驟,所以可以配對
(`docs/cc4d_format_questions.md`)。

手動排除:太模糊、評審無法判斷的參考答案。先篩出字數 ≤ 3 或用了籠統詞("improperly"、
"wrong")的描述,再人工逐條看。這個篩選不保證找出全部。

| recording | step_id | 參考答案 |
| --- | --- | --- |
| 1_34 | 9 | The top piece |
| 2_28 | 17 | Not properly peeling the onion. |
| 2_30 | 17 | Peel improperly |
| 2_46 | 14 | Peeled the garlic improperly |
| 5_28 | 71 | The water temperature is wrong |
| 5_42 | 60 | Not properly placing the paper filter. |
| 16_40 | 176 | Not properly peeling the Garlic cloves. |
| 16_40 | 180 | Not properly peeling the Garlic cloves. |
| 18_28 | 200 | Peel improperly |
| 20_26 | 176 | Peel garlic cloves improperly |
| 20_39 | 176 | Improperly peel the garlic cloves. |
| 26_24 | 303 | Used wrong shaped mug |
| 28_50 | 330 | Rolling the lettuce wraps in a wrong way. |

## 要報告的指標

1. **欄位準確率:** 整體、各題型、各欄位類別。欄位類別的例子:`stepNN_correct`、
   `stepNN_duration`、`skipped_steps`。CC4D 的欄位類別是欄位 id 去掉 `stepNN_` 前綴。
2. **各欄位類別的多數答案基準線:** 一律回答最常見答案的準確率,放在模型準確率旁邊。例如:在
   CC4D 上,`stepNN_correct` 一律答 `true`,在真實步驟上約有三分之二答對(確切數字在資料集
   README)。
3. **次要測量:**
   - `int` / `seconds`:`abs_error` 的中位數,`seconds` 另外報 `rel_error` 的中位數。
   - `multi_choice`:`hits` / `misses` / `extras` 的加總。
   - `open_text`:`extra_claims` 的加總,以及人工標註樣本上的評審一致性。

   只計算有效、具體的答案。
4. **跳過步驟的漏答(`cc4d_dense_v1` 的幻覺檢查):** `cc4d_dense_v1` 沒有假步驟,所以用
   `skipped_steps` 檢查。模型沒有回答的答案標籤,就是一個沒做、但模型當作有做的步驟。這些
   `misses` 和比例 `misses / (hits + misses)` 要單獨報告,和 `extras`(有做卻被說成跳過的步驟)
   分開。只計算有效答案。
5. **格式失敗:** 各題型的 `invalid_answer` / `missing_answer` / `invalid_output` 數量。
6. **跨題型比較準確率。** 不同題型的原始準確率不能直接比,因為隨機猜中的機率差很多:`bool` 50%、
   8 個選項的 `single_choice` 12.5%、7 個選項的 `multi_choice` 要完全答對接近 0、`int` /
   `seconds` 則取決於容許誤差。所以:
   - 每個準確率格子(題型 × 內容 × 條件)旁邊都報告**隨機基準線**和**多數答案基準線**(第 2 點)。
   - 題型效果用**配對差異**衡量:同一個事實在不同 call 裡用不同題型問,逐題比較。
   - 必須直接比較格子時,另外報告扣除隨機基準線的準確率:`(acc − chance) / (1 − chance)`。
     `int` / `seconds` 的 chance 是在相同容許誤差下,一律回答多數答案基準線那個值的命中率。
7. **選項固定的 `multi_choice` 欄位逐選項分析**(CC4D 的 `stepNN_error_types`,選項是 7 個錯誤
   標籤)。每個選項報告:
   - **recall:** 答案含有這個選項的欄位中,回答也含有它的比例。衡量漏掉的錯誤。
   - **precision:** 回答含有這個選項的欄位中,答案也含有它的比例。衡量誤報。
   - 兩個數字背後的計數。

   只用有效、具體的答案。例如:`Order` 的 recall 20%、`Technique` 的 recall 70%,表示模型看得到
   灑出來,但看不出順序錯誤。只看加總的 `hits` / `misses` / `extras` 會看不出這一點。選項每份報告
   都不同的欄位(例如 `skipped_steps`)只報告加總。
