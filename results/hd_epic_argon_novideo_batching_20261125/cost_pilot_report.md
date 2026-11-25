# HD-EPIC Argon 無影片費用試跑（2026-11-25）

20/20 成功，0 失敗；四種 k 各 5 次，完整 prompt 與生成設定保留，只移除影片。原始回應逐筆重新解析，所有答案 key 完整且為 A–E。

試跑合計優惠費率估算 US$0.205064，未抵扣等值 US$0.410128。非實際帳單。

以下 token 皆為每次呼叫平均；answer 與 thinking 分開列示：

- k=4：input 1243.8、answer 55.6、thinking 78.2 tokens；單次 US$0.0038256；全部 584 次推算 US$2.2342。
- k=8：input 2255.2、answer 109.0、thinking 0.0 tokens；單次 US$0.0056004；全部 292 次推算 US$1.6353。
- k=16：input 4288.2、answer 211.2、thinking 0.0 tokens；單次 US$0.0106884；全部 146 次推算 US$1.5605。
- k=32：input 8362.2、answer 417.4、thinking 0.0 tokens；單次 US$0.0208984；全部 73 次推算 US$1.5256。

全部 1,095 次（含成功試跑）推算 US$6.9556；餘下 1,075 次推算 US$6.7505。未抵扣費率為兩倍。此為各 k 五個選定影片的平均外推，沒有費用信賴區間，thinking 長度可能有長尾。若以每個 k 在這五次中最高單次費用外推，全部約 US$8.8231；這仍不是上限。

建議正式批次採優惠費率估算 US$10 停止線（包含本次試跑費用，不包含前一關 k1 試跑），未抵扣等值 US$20。這是本地估算與預留費用的停止線，不是雲端硬性帳單上限。遇到失敗／不確定呼叫就停下，不自動重試。

正式 runner 已離線驗證：

```sh
python argon_novideo/full_run.py
```

待使用者同意正式預算後才執行：

```sh
python argon_novideo/full_run.py --execute --max-estimated-usd 10 --workers 4
```

會重用已成功的20個試跑回應，只補1075次；不啟動影片漂移檢查、參考題或其他實驗。4個worker僅調整同時呼叫數，不改變單次題目分組、順序、prompt或生成參數。正式原始回應路徑為 results/hd_epic_argon_novideo_batching_20261125/。目前尚未建立此結果目錄、未送出正式呼叫。

證據：本目錄 calls/ 為原始回應；per_field.csv 為300題逐題結果；cost_summary.csv 為各k費用與token明細；completion_audit.json 為獨立重解析與原始檔hash。5個單元測試通過，正式runner預設離線驗證通過。

k8/16/32的五次回應 usageMetadata 回報 thinking tokens=0；k4為391。已確認所有送出 body 都指定MEDIUM，但無法由usage獨立觀察供應商內部實作。不要因thinking=0推論設定未送出。

準確率不在此5部影片費用試跑中做科學結論；等完整配對資料收齊再分析。正式跑仍待使用者預算同意。
