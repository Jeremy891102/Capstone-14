# Qwen3-VL-32B 注意力干預:放大影片注意力能補救 k = all 嗎?

狀態(2026-11-21):完成,涵蓋 54 份報告(A 類 23、B 類 31),k = all、正序。

模型:Qwen3-VL-32B-Instruct。資料:`cc4d_k32_v1`。背景與促成這個實驗的注意力量測見 [attention_probe.zh.md](attention_probe.zh.md)(k = all 的晚層影片注意力比 k = 8 約少 13 pp)。

## 1. HF 生成計時測試(實驗前的成本評估)

干預實驗要用 HF 逐 token 生成,所以先量成本,並檢查 HF 的輸出是否和 vLLM 當時存下的一致。設定:4 張 A100 80GB,bf16,greedy,MLP 分段計算(`--mlp-chunk 32768`),每張卡放 22 GiB 權重。輸入與注意力量測相同(見 [attention_probe.zh.md](attention_probe.zh.md) 第 2 節)。

| prompt token | 輸出 token | 預填 | 解碼 | 總時間 | GPU 0 峰值 | 與 vLLM 輸出 |
| --- | --- | --- | --- | --- | --- | --- |
| 57,669(A) | 293 | 41 s | 11.3 tok/s | 67 s | 30.9 GiB | 27/31 欄位相同 |
| 86,574(A) | 389 | 78 s | 9.7 tok/s | 118 s | 37.3 GiB | 37/39 欄位相同 |
| 155,653(A,最長) | 535 | 213 s | 7.3 tok/s | 286 s | 55.0 GiB | **55/55,逐字相同** |
| 211,113(B) | 573 | 371 s | 6.1 tok/s | 464 s | 69.3 GiB | **61/61,逐字相同** |

- **輸出和 vLLM 高度一致,但不保證逐字相同:** 兩個較短的呼叫各有 4 與 2 個欄位的答案不同(推測是兩邊 kernel 與 MLP 分段計算的 bf16 捨入差異,讓貪婪解碼在某處分岔,未驗證)。所以干預實驗的基準要用 HF 自己的 α = 1,不能直接拿 vLLM 的結果當對照。
- **成本:** 總時間對 prompt 長度呈超線性(預填的注意力隨長度平方成長)。用前三點擬合 `t ≈ 0.764 N + 0.00689 N²`(N 單位 k token,t 單位秒),預測 211k 為 468 s,實測 464 s。照此估算 k = all 一次完整生成(單一方向、單一條件):A 類 23 份約 1.3 小時,B 類 31 份約 3.4 小時,合計約 **4.7 小時**。每個 α、每個對照、每個方向都要各乘一次。
- **記憶體:** GPU 0 峰值隨 prompt 長度線性成長(約 0.26 GiB 每千 token)。B 類最長的 prompt 約 23.9 萬 token,外插約 76 GiB,逼近單卡上限(約 79 GiB 可用),**最長的幾次可能放不下**,要實測。

## 2. 干預實驗:方法與結果

對 54 份報告的 k = all 答案重新生成(`scripts/qwen_intervene.py`,評分 `results/cc4d_intervene_score.py`)。預填一次,之後每個條件從同一份 KV cache 解碼。只對正在生成的 token,在 softmax 前對目標 key 的注意力分數加 log α;第一個生成 token 來自預填,不受影響。

條件:`base`(α = 1,同一條手動注意力路徑);`a1.5`、`a2`、`a3`(最後 1/4 的層,影片 key);`all2`(全部層,影片 key);`text2`(對照:最後 1/4 的層,prompt 文字 key)。基準用 `base` 而不是 vLLM 存下的答案,因為手動注意力與融合 kernel 差 bf16 捨入;兩者準確率一致(33.4% 對 33.5%)。

2,557 題一般題、54 份報告,95% 區間為對報告 bootstrap:

| 條件 | 準確率 | 對 base 差(pp)[95% 區間] | 晚層影片注意力 | 答案與 base 不同的比例 |
| --- | --- | --- | --- | --- |
| base | 33.4% | – | 49.6% | – |
| a1.5 | 33.6% | +0.2 [−0.1, +0.6] | 57.6% | 3.2% |
| a2 | 33.0% | −0.4 [−1.2, +0.2] | 62.8% | 4.7% |
| a3 | 32.7% | −0.6 [−1.6, +0.1] | 69.4% | 6.0% |
| all2 | 33.3% | −0.0 [−0.6, +0.5] | 62.5% | 7.4% |
| text2(對照) | 33.0% | −0.4 [−0.7, −0.0] | 36.0% | 4.4% |
| vLLM k = 8(參考) | 35.6% | +2.3 [−0.2, +4.7] | – | – |

- 干預有效(影片注意力隨 α 上升,文字對照則下降),α = 3 的影片注意力甚至高於 k = 8 的水準(60%),但準確率沒有往 k = 8 移動。
- 各題型沒有一致方向(bool、multi_choice、seconds 都在 ±1 pp 內;整數與單選分別只有 54、39 題)。
- 文字對照也讓準確率略降,所以 α = 2–3 的小幅下降同樣可以解釋成「擾動注意力本身有點傷害」。
- 輸出全部可解析,長度不變。
- 解讀:[attention_probe.zh.md](attention_probe.zh.md) 3.2 節的 13 pp 注意力差距不是 k = all 準確率較低的原因,至少用單一全域倍率補回來沒有幫助。限制:只在晚層用同一個倍率,沒有逐 head 干預;k = 8 與 k = all 只差 2–4 pp,偵測空間小;α 最高只到 3。

## 3. 重現

- `scripts/qwen_intervene.py RUN_DIR OUT.jsonl --model ... --conditions base,a1.5,a2,a3,all2,text2 --mlp-chunk 32768 --max-gpu-mem 22`(A 類加 `--max-seconds 660`;B 類加 `--min-seconds 660 --max-seconds 1020`)。可續跑。在 vm2 上用 `~/envs/capstone-qwen8b`、4 張 A100 80GB、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;佇列腳本 `~/intervene_queue.sh`,進度 `~/intervene_queue.log`。
- `results/cc4d_intervene_score.py RUN_DIR GT.jsonl ROWS.json.gz OUT.json INTERVENE.jsonl ...`:用專案的 parser 與 scorer 評分。
- 原始輸出(只在 vm2,不在 repo):`~/cc4d_run/gen/intervene_A.jsonl`(138 筆)、`intervene_B.jsonl`(186 筆),評分摘要 `gen/intervene_scores.json`。每個 (呼叫, 條件) 一筆:生成文字、注意力佔比、token 數、時間。
- 計時測試的腳本與結果(`~/cc4d_run/gen_timing.py`、`gen/timing.jsonl`)只在 vm2 上,還沒放進 repo。
