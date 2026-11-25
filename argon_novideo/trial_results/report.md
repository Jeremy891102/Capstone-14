# k=1 subset10 對照試跑（2026-11-25）

320/320 成功，0 失敗，0 無效 JSON，modelVersion 全為 gemini-4-argon。逐筆重新解析保存的回應，核對歷史請求、target 與評分。

本次 84/320 = 26.25%；歷史 79/320 = 24.6875%；差 +1.5625 pp，未超過事前約 5 pp 的查錯門檻。

逐題答案一致 179/320 = 55.9375%，141 題答案改變。總準確率相近，不代表逐題作答穩定；沒有設定 temperature 或 seed，API 預設採樣與跨時間供應商變化的影響無法由本次對照拆開。未因準確率接近就宣稱排除模型漂移。

使用量：155,115 input、5,602 answer、2,104 thinking tokens。依已查閱優惠費率估算 US$0.38729，未抵扣費率等值 US$0.77458；非實際帳單。API 往返累計約 1,438.61 秒，與實驗經過時間不是同一指標。沒有重試或不確定帳單呼叫。

完整請求保存於 frozen/k1_subset10.jsonl；每次回應保存於本目錄 calls/；逐題比較見 per_field.csv；稽核含原始回應 hash 見 completion_audit.json。

每個 request body 仍為 MEDIUM、maxOutputTokens=32768、JSON；system/user 及影片標籤文字逐字保留，只移除影片。Wire 設定與 thinking token 回傳可確認，無法獨立觀察供應商內部是否完整遵守 MEDIUM。

第一關已停止，未啟動 k4/8/16/32 費用試跑或正式實驗。下一步：使用者審查本結果後，準備並執行每個 k 各 5 次、共 20 次費用試跑，回報 token 與全跑估算；正式實驗仍需預算批准。
