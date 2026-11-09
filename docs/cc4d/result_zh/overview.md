# CC4D 答案題型實驗:結果總覽

實驗設計見 [cc4d_answer_type_proposal.md](../cc4d_answer_type_proposal.md) 與 [cc4d_format_questions.md](../cc4d_format_questions.md)。本文只列六個研究問題的主要結論,細節、表格與圖在各自的文件。

## 資料與模型

| 模型 | 範圍 | 備註 |
| --- | --- | --- |
| Gemini 3.5 Flash-Lite | 完整(41 組,25,711 次呼叫) | 大多接近亂猜 |
| Qwen3-VL-32B | 完整(同一題組);open_text 於 11/8 補評分 | 長影片解析度被降低 |
| Gemini 4 Argon | 只有 base 組的 5 格(duration bool / single、localization bool / single / seconds),k = 1 / 4 / 8 與倒序 k = 8 | 沒有 error、沒有 E4、沒有幻覺題、沒有 E5 |

原始結果在 `results/cc4d_answer_types_v1_*`;本批分析的腳本與輸出在 `results/cc4d_q1_*`、`cc4d_q2.py` 到 `cc4d_q6.py` 與對應的 `*_out.txt`。圖在 [figs/](figs/)。

## 主要結論

| 問題 | 主要結論 | 詳細文件 |
| --- | --- | --- |
| **Q1** 每次 call 問更多題,準確率會不會變?會不會因題型不同? | **看模型,不看題型。** Qwen 隨題數下降(k = 1 → 8 約 −4.6 pt),集中在 localization 的 bool 與 single,k = 16 時 localization 掉約 10 pt;Flash-Lite 與 Argon 沒有可靠的題數效應。題數效應隨內容改變(三向交互 p = 0.0002),不隨題型改變(duration + localization 的三向交互 p = 0.85)。Qwen 的下降一部分來自 bool 越來越偏向答「是」 | [q1.md](q1.md) |
| **Q2** 同一個事實換題型問,準確率會不會變? | **會,但主要是答案偏誤,不是能力。** error 內容裡所有題型的準確率幾乎全來自「沒有錯」的步驟,有錯的步驟答對率都不到 33%;換題型改變的是模型說「沒有錯」的傾向。各模型的題型排名不同(模型 × 題型交互 p < 0.0001);共同點是 localization 的 single 比 bool 好、duration 的 seconds 比 bool 差 | [q2.md](q2.md) |
| **Q3** 模型在每種內容上表現如何? | **Qwen 與 Argon:localization 最高、duration 其次、error 最低;Flash-Lite 在三種內容都接近亂猜,看不出差別。** 用 bool 平衡準確率確認差異是真的分辨力。此結論只是描述性的,三種內容的題目不是同一批事實 | [q3.md](q3.md) |
| **Q4** 混合題型會不會影響同一批題目? | **不會變差,反而小幅變好**(Flash-Lite 與 Qwen 都約 +2 pt),集中在 bool 題,兩個模型一致。位置相同的配對才是乾淨的檢定;位置改變本身對 Qwen 有影響(往後移掉 4 pt)。進步原因不明,不是答案偏誤改變造成的 | [q4.md](q4.md) |
| **Q5** 題型和題數會不會影響幻覺?沒做的步驟在 call 裡的位置有關嗎? | **題型影響很大且各模型排名不同;題數的影響方向相反**:Flash-Lite 幻覺率下降(66% → 50% → 52%)但更常把有拍的步驟說成看不到(2.9% → 6.4%),Qwen 小幅上升(70% → 74% → 77%)。沒做的步驟在 call 裡的位置沒有明確效應;放進沒做的步驟對其他題的影響在 ±1.1 pt 內 | [q5.md](q5.md) |
| **Q6** 題目的位置(順序)會不會影響準確率? | **只有 Qwen 受影響:倒序讓它變差,題數越多越嚴重**(k = 8 約 −4 pt,k = 16 的 localization 約 −19 pt)。不是「排在後面比較差」,是倒序本身;Flash-Lite 與 Argon 不受影響 | [q6.md](q6.md) |

## 跨問題的共同發現

1. **答案偏誤比能力差異更常見。** Flash-Lite 與 Qwen 在 error 內容都偏向答「沒有錯」(Q2),Qwen 的 bool 隨題數越來越偏向答「是」(Q1),題數增加時 Qwen 的 open_text 準確率上升,是因為更常答「none」(Q1);Flash-Lite 的 int 上升我推測是同一類現象,但沒有驗證。看準確率變化時,需要同時看有錯與沒錯、真答案是與否的拆分。
2. **Qwen 對 call 的格式最敏感。** 題數增加(Q1)、倒序(Q6)、位置往後移(Q4)都讓它變差,而且集中在 localization 與 duration,也就是它最強的內容;兩個 Gemini 模型在已測範圍內都沒有可靠的題數或順序效應。
3. **混合題型不傷害準確率**(Q4),所以同一個 call 裡放不同題型,對 Flash-Lite 與 Qwen 是安全的。
4. **Flash-Lite 靠更保守來降低幻覺**(Q5):題數增加時更常說看不到、更常說沒有錯,幻覺率下降,但誤判沒做上升。
5. **三種內容的難度排序**(Q3)在 Qwen 與 Argon 一致,與 Qwen 在 Q1 的下降位置(localization)重疊,所以「題數 / 順序的影響」與「內容難度」不能簡單分開看。

## 共通限制

- **Argon 只有 5 格**,沒有 error、E4、幻覺題、E5,所以 Argon 的「沒有題數 / 順序效應」只對 duration 與 localization 成立,而且它的 localization 接近上限(95 到 97%),偵測下降的能力有限。
- **Flash-Lite 大多接近亂猜**,看不出效應不等於沒有效應。
- **所有檢定都沒有多重比較校正**,p 值在 0.01 到 0.05 之間的結果只當線索。
- **內容與題型沒有完全交叉**,所以多數分析分區塊做,不是一個全因子模型;三種內容的題目不是同一批事實,內容比較只能是描述性的。
- **Qwen 的長影片解析度被降低**,與其他兩個模型不是同一個條件。
- **評分規則有事後調整的部分**:Flash-Lite 的 `typed_v3`(例如把 1004 讀成 10:04)是看過結果後才定的;open_text 的評審與 100 題人工標註一致率 78%,偏嚴。
- **有些原因還沒找到**:Q4 的 bool 進步、Q6 的 Qwen 倒序下降(只確認有一部分是答案差一格錯位),都只是描述。
