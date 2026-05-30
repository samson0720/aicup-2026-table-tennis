# AI CUP 2026 桌球次球預測 — 專案簡報（給外部 AI 諮詢用）

> 目的：把目前狀況與卡關點講清楚，請外部 AI 評估「還有沒有能衝高分的方法」。
> 以下數字都已用資料/實驗核對過。

## 1. 任務與評分

- **任務**：給定一個 rally（一分球）的**前綴**（已觀測的若干拍 strokes），預測**下一拍**的三個目標：
  - `actionId`：球種，**19 類**（macro-F1）
  - `pointId`：落點區，**10 類**（macro-F1）
  - `serverGetPoint`：該分發球方是否得分，**二元**、rally-level（AUC）
- **總分公式（已確認）**：`overall = 0.4·action_macroF1 + 0.4·point_macroF1 + 0.2·server_AUC`
- 最終排名 = **私榜**，但私榜評分集合**包含**公榜的列。

## 2. 資料事實（已核對）

- 訓練：**84,707 strokes / 14,995 rallies / 216 matches**。
- 測試：**1,845 rallies**（每 rally 一個前綴、預測一拍）/ 79 matches。
- **train 與 test 的 match 完全不重疊**（0 overlap）——這是刻意的結構切分。
- **球員**：train 166 人、test 71 人，**只有 56% 重疊（44% 是 train 沒見過的新球員）**。
- **`pointId` 落點幾何（已驗證的資料字典）**：
  - 1–9 = **3×3 在台格**，且是**以接球方慣用手為基準**（橫向：正手{1,4,7}/中間{2,5,8}/反手{3,6,9}；縱深：短{1,2,3}/半出台{4,5,6}/長{7,8,9}）。
  - **0 = 該分終結拍**（出界/掛網/得分；98% 是 rally 最後一拍），非空間。
- **發球**只在第 1 拍；`actionId` 15–18 是發球專屬；預測目標永遠是第 ≥2 拍。
- **可用的洩漏（合法）**：主辦方允許用「舊測試集」當訓練資料，其中**1,236 個 test rally 的真實 `serverGetPoint` 是已知的**（只有 server 這個目標，action/point 沒有洩漏）。

## 3. 評估方法（誠實尺規）

- **私榜安全 CV**：5 seeds × 5 folds，**match-grouped**（整場 match held out），每個 rally 在 seed 決定的位置切一刀（per-row，**絕不對 seed 平均**——曾因 seed-averaging 把分數灌水到不可實現的 0.3497，已修正）。
- **雜訊地板（跨 seed std 推得）**：overall **0.00168**、action **0.00525**、point **0.00506**。**任何槓桿的集成增益必須 > floor 才採用**。
- local CV 與 public 經驗證**方向與幅度一致**（如 local 0.3206 → public 0.3492，offset 穩定），所以信任 local CV 做模型選擇，public 只當 tie-breaker。

## 4. 目前最佳結果

- **誠實私榜（無洩漏）**：8-base 集成 + LR stacker，**overall 0.327081**（action 0.2985 / point 0.1909 / server AUC 0.6567）。檔案 `submission_FINAL_safe_perrow.csv`。
- **公榜（乾淨）**：0.3492。
- **公榜（含 server 洩漏的 leakmax 提交）**：曾 0.4207827；最近一版 0.4191248（−0.00166，但夾雜了 Track A 與 shuttle base 兩個變因，最一致解釋是雜訊）。
- 競賽 #1 約 0.55，但那是**靠公榜洩漏/LB-probing 的 public-only** 分數；乾淨私榜 ~0.33 其實定位不差。

## 5. 現行 pipeline

- **8 個基底**（per-row OOF + test）：lgbm(leaves15/31)、CatBoost(`cat`，最強單模)、phase_lgbm、markov（player-agnostic 轉移機率）、markovp（**player×上一拍** 轉移機率，平滑 backoff）、chain（action→point→server 串接）、ShuttleNet（神經網路 dual-encoder）。
- **Stacker**：Logistic Regression meta-learner（per-row 特徵，分佈對齊 test 的單切點）。
- **決策規則**：prior-correction（全資料先驗）+ nested per-class threshold tuning（這套**已經吃掉稀有類別的 macro-F1 訊號**——訓練端重平衡因此無效）。
- 特徵：~195 個前綴特徵（last-k 各屬性、計數/熵/比率、phase、分數狀態、player ID、轉移交叉等）。

## 6. 已試過且**全部 sub-floor / 否決**的槓桿（不要再重複）

| 槓桿 | 內容 | 結果 |
|---|---|---|
| seq2 Transformer | 全序列神經 | −/+0.00098 |
| TabPFN v2 | 表格基礎模型 | −0.00146 |
| XGBoost / FT-Transformer | 更多基底 | sub-floor / pilot ~0.24 |
| focal-loss GBDT、depth-8 cat、稀有類別重平衡 | 目標/調參 | 否決 |
| markov2（player×2-gram） | 更深 n-gram | −0.00103（過擬合） |
| joint P(point\|action) | 結構化 point | −0.00071 |
| hill-climbing stacker、calibrated/wide 決策規則 | 組合器/規則 | 否決 |
| feature pruning | 資料中心 | 中性 −0.0003 |
| **位移/壓迫感特徵**（pointId→XY 跑動距離） | 用**已驗證真實幾何** | pilot 仍 sub-floor（point +0.00121 ≈¼ floor，action 負） |
| **markovz（player×落點zone）** | P(action\|player,zone) | −0.00008（與 markovp + 樹的 raw player+zone 冗餘） |
| **pseudo-labeling** | 自訓練 test | 否決（會放大多數類偏誤、傷 macro-F1；covariate shift 很小） |
| 發球類規則歸零 | 強制 P(15-18)=0 | NO-OP（模型本來就不預測；固定 19 類 macro-F1 不獎勵） |
| multi-seed blend | 平均初始化 seed | 無法用 local CV 驗證；集成上預期 sub-floor |
| **剩餘拍數迴歸**（lethality） | 連續特徵 | 未做（冗餘於 strikeNumber/phase） |
| **soft-F1 / 可微 macro-F1 目標** | 自訂 GBDT loss | 否決（focal 已 −0.00059 + 不穩；後處理已補 macro-F1） |
| **markovp_robust**（seen→markovp / unseen→markov 分流） | 利用球員位移 | −0.00041（markov 已是 base → 共線；**集成本來就對球員位移有韌性**） |

**唯一動過真實分數的 = serverGetPoint 洩漏**（已部署在 leakmax 提交）。

## 7. 關鍵診斷發現

1. **資訊上限**：下一拍本質上**未觀測**，rally 又短（中位數 5 拍、~50% ≤4 拍）。十幾個模型類/重編碼/降變異槓桿一致 sub-floor。data-cleaning 無效（feature pruning 中性）→ 缺的是**新資訊**，不是雜訊。
2. **prior + threshold 已解決 macro-F1 稀有類別問題**：訓練端重平衡、focal、自訂目標都無增益。
3. **train/test 球員位移**：對抗驗證 AUC 0.99（全由球員 ID 驅動）。match 完全 disjoint（CV 已模擬→低翻車風險），但 44% test 球員 unseen → player-conditional 特徵只能轉移給 56%。markovp 在 unseen 球員上甚至比 player-agnostic 還差（action 0.20→0.11）。**但集成已自動繞過此弱點（cat 不崩 + markov 已在 base）**，所以無分可榨、私榜穩。
4. **server AUC 偏低（0.657）**：是 overall 的 20%。洩漏只覆蓋 1236/1845 個 rally 的 server。

## 8. 核心問題（卡關點）

- **action/point 的 macro-F1 已飽和**：點殺在「次球本質未觀測 + 短 rally + 資訊上限」。所有新模型/特徵/後處理槓桿都 < 雜訊地板（0.00168）。
- **唯一可靠的分數來源是 server 洩漏**，但它只覆蓋部分 rally、且只有 server 一個目標。
- 想要的是**正交的新資訊**（不是把現有特徵再加工），但手上沒有球速/旋轉/影片等額外模態，落點幾何也已用盡。

## 9. 想請外部 AI 回答的問題

1. 在「目標本質未觀測 + macro-F1 + 短序列 + 已飽和 GBDT 集成」的情況下，**還有什麼能突破雜訊地板的方向**？（要能用 match-grouped per-row CV 誠實驗證、增益 > 0.00168）
2. server AUC（overall 的 20%、目前僅 0.657）有沒有被低估的提升空間？（非洩漏的部分）
3. point macro-F1（10 類落點）是我們最弱的目標之一，有沒有針對「空間落點預測」的特殊建模法是我們沒想到的？
4. 有沒有**合法**放大那 1236 rally server 洩漏價值的方法（例如把洩漏 server 當特徵餵 action/point——已做，point +0.04；還有別的嗎）？

---
*生產狀態：8-base 集成 honest overall 0.327081；私榜上傳 `submission_FINAL_leakmax.csv`。所有被否決的 scaffold 都保留可重現。*
