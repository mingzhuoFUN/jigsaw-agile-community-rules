# Jigsaw Agile Community Rules

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mingzhuoFUN/jigsaw-agile-community-rules/blob/main/notebooks/verified_ettin_colab.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

闈㈠悜绀惧尯瑙勫垯鐞嗚В鐨勬枃鏈垎绫荤郴缁熴€傞」鐩皢璇勮銆佺ぞ鍖恒€佽鍒欐枃鏈強姝ｅ弽绀轰緥缁勭粐涓虹粺涓€杈撳叆锛岄€氳繃鍙潬鐨勯獙璇佽璁°€佺█鐤忕壒寰佸熀绾夸笌 Transformer 妯″瀷杈撳嚭杩濊姒傜巼銆?
## 椤圭洰姒傝

| 椤圭洰 | 鍐呭 |
|---|---|
| 浠诲姟 | 鍒ゆ柇 Reddit 璇勮鏄惁杩濆弽缁欏畾绀惧尯瑙勫垯 |
| 杈撳叆 | `body`銆乣subreddit`銆乣rule`銆佹渚嬩笌鍙嶄緥 |
| 杈撳嚭 | 姣忎釜 `row_id` 鐨?`rule_violation` 姒傜巼 |
| 璇勪及鎸囨爣 | ROC AUC |
| 鏍稿績闅剧偣 | 娴嬭瘯闆嗗寘鍚缁冮樁娈垫湭鍑虹幇鐨勮鍒欙紝闇€瑕佸熀浜庤鍒欒涔夋硾鍖?|
| 鎺ㄨ崘鍏ュ彛 | [鍦?Google Colab 涓繍琛?Ettin 妯″瀷](https://colab.research.google.com/github/mingzhuoFUN/jigsaw-agile-community-rules/blob/main/notebooks/verified_ettin_colab.ipynb) |

## 绯荤粺娴佺▼

```mermaid
flowchart LR
    A["璇勮涓庣ぞ鍖轰俊鎭?] --> E["缁熶竴鏂囨湰琛ㄧず"]
    B["瑙勫垯鏂囨湰"] --> E
    C["杩濊绀轰緥"] --> E
    D["鍚堣绀轰緥"] --> E
    E --> F{"寤烘ā璺嚎"}
    F --> G["TF-IDF + Logistic Regression"]
    F --> H["Ettin / Transformer"]
    F --> I["澶氭ā鍨嬮泦鎴?]
    G --> J["浜ゅ弶楠岃瘉涓庢鐜囨牎鍑?]
    H --> J
    I --> J
    J --> K["rule_violation 姒傜巼"]
    K --> L["submission.csv"]
```

## 鏁版嵁姒傝

| 鏁版嵁闆?| 琛屾暟 | 鍒楁暟 | 璇存槑 |
|---|---:|---:|---|
| `train.csv` | 2,029 | 9 | 甯?`rule_violation` 鏍囩 |
| `test.csv` | 10 | 8 | 鍏紑娴嬭瘯鏍蜂緥 |
| `sample_submission.csv` | 10 | 2 | 鎻愪氦鏍煎紡 |

璁粌鏍囩鍒嗗竷杈冨潎琛★細姝ｇ被 1,031 鏉★紝璐熺被 998 鏉°€傝缁冮泦鍖呭惈 `No legal advice` 涓?`No Advertising` 涓ょ被瑙勫垯锛屾ā鍨嬮渶瑕佸埄鐢ㄨ鍒欐枃鏈笌绀轰緥瀹屾垚璺ㄨ鍒欐硾鍖栥€?
## 寤烘ā鏂规

| 妯″潡 | 瀹炵幇 |
|---|---|
| 鏂囨湰鏋勯€?| 璇勮銆佺ぞ鍖恒€佽鍒欍€佹鍙嶇ず渚嬪垎鍖烘嫾鎺?|
| 绋€鐤忕壒寰?| word 1鈥? gram + `char_wb` 3鈥? gram TF-IDF |
| 鍩虹嚎妯″瀷 | Logistic Regression + sigmoid calibration |
| 娣卞害妯″瀷 | Ettin-400M 缂栫爜鍣?|
| 楠岃瘉 | Stratified K-Fold AUC锛屽苟鎵╁睍瑙勫垯/绀惧尯鐣欏嚭楠岃瘉 |
| 闆嗘垚 | 澶氭ā鍨嬮娴嬫寜 `row_id` 瀵归綈鍚庡姞鏉冭瀺鍚?|

## 蹇€熷紑濮?
### Google Colab

鐐瑰嚮涓嬫柟鎸夐挳鍗冲彲鎵撳紑宸查厤缃殑杩愯鐜锛?
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mingzhuoFUN/jigsaw-agile-community-rules/blob/main/notebooks/verified_ettin_colab.ipynb)

鍦?Colab Secrets 涓坊鍔?`KAGGLE_API_TOKEN`锛屽苟纭 Kaggle 璐﹀彿宸叉帴鍙楃珵璧涜鍒欍€?
### 鏈湴鐜

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

涓嬭浇绔炶禌鏁版嵁锛?
```powershell
.\download_data.ps1
```

杩愯鏁版嵁鍒嗘瀽涓庡熀绾胯缁冿細

```powershell
$env:PYTHONIOENCODING='utf-8'
python src/eda.py
python src/train_baseline.py
```

涓昏杈撳嚭锛?
- `outputs/submission_baseline.csv`
- `outputs/oof_baseline.csv`
- `outputs/metrics_baseline.json`

## 椤圭洰缁撴瀯

```text
notebooks/                 # Colab 璁粌鍏ュ彛
configs/                   # 妯″瀷涓庤缁冮厤缃?scripts/                   # Notebook 鏋勫缓鍙婅缁冪紪鎺?src/
  first_place/             # 鏁版嵁鏋勯€犱笌闆嗘垚妯″潡
  eda.py                   # 鎺㈢储鎬ф暟鎹垎鏋?  train_baseline.py        # TF-IDF 鍩虹嚎
tests/                     # 鏁版嵁銆佹寚鏍囦笌铻嶅悎閫昏緫娴嬭瘯
```

## 宸ョ▼浜偣

- 灏嗚鍒欐枃鏈笌姝ｅ弽绀轰緥浣滀负涓€绛夎緭鍏ワ紝鏀寔鏈瑙勫垯娉涘寲銆?- 鍚屾椂鎻愪緵杞婚噺鍩虹嚎涓?GPU 娣卞害妯″瀷璺緞锛屼究浜庡揩閫熼獙璇佸拰鎵╁睍銆?- 棰勬祴鎸?`row_id` 涓ユ牸瀵归綈锛岄檷浣庡妯″瀷铻嶅悎鏃剁殑鏁版嵁閿欎綅椋庨櫓銆?- Colab銆並aggle Token銆丟oogle Drive 杈撳嚭璺緞褰㈡垚瀹屾暣浜戠璁粌閾捐矾銆?
## 鍚庣画鏂瑰悜

- 鎸夎鍒欐垨绀惧尯杩涜鐣欏嚭楠岃瘉锛屾洿璐磋繎闅愯棌娴嬭瘯鍒嗗竷銆?- 鍒嗗埆缂栫爜璇勮銆佽鍒欎笌绀轰緥锛屽苟鍔犲叆璇箟鐩镐技搴︾壒寰併€?- 瀵?TF-IDF銆乀ransformer 涓庤鍒欑壒寰佽繘琛?rank averaging銆?- 澧炲姞妯″瀷璇樊鍒嗘瀽涓庡垎瑙勫垯 AUC 鍙鍖栥€?
