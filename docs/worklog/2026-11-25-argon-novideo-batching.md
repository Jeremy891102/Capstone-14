# HD-EPIC Argon 無影片 batching（2026-11-25）

使用者批准執行k1 subset10對照、20-call費用試跑與k4/8/16/32正式實驗，並明確批准commit/push。完整prompt逐字保留，只移除影片。

正式1095個接受結果完整，含20個費用試跑重用；另重用2336個完整歷史k1原始回應。1次HTTP500 INTERNAL封存並人工覆核後補送，成功呼叫不重跑。

無影片k1/4/8/16/32為26.54/30.95/32.28/35.36/37.80%。無影片k8−k1 +5.74pp，95%CI[3.13,8.48]；影片增益k8−k1 +0.68pp，CI[-2.18,3.55]；影片增益k32−k8 -10.70pp，CI[-13.74,-7.71]。有影片與無影片並非同時期收集，不將協定交互作用視為內部視覺機制的因果證明。

新增已知成功用量估算共$7.181044（含另做subset10 $0.387290），一次500失敗帳單未知且不包含在估算。7個測試通過，原始回應重解析、用量獨立重算、逐題配對、pilot逐位元重用與圖像檢查皆完成。

完整證據：[README](../../results/hd_epic_argon_novideo_batching_20261125/README.md)、summary.json、completion_audit.json、raw_response_hashes.json與PNG/SVG圖。程式與試跑證據在argon_novideo/。沒有改動其他工作、既有結果、套件或GPU工作。
