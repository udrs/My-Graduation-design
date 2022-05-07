# Two-Stream-BRDF-Network_Reproduction
オープンソースではなく「Highlight-aware Two-stream Network for Single-image SVBRDF Acquisition」という論文のリプロダクション

リプロダクションの過程は三つに分かれる。

## １．DataSetの処理
## ２．Network　Modelの構築
### 2.1 STとHAのTwo-Stream　Network
### 2.2 Outputは四つなので、2.1に合弁されたfeature　mapは四つのFU　Networkに分かれた。
### 2.3 予測値の精度を上がるために、perceptual adversarial lossを使って、 two global context discriminators　and two local context discriminatorsをトレーニングする。
## ３．Ground　TruthとLableに基づいてのLoss関数
### 3.1 SVBRDF_Loss関数 
### 3.2 Rendering_Loss関数（それを使ったら、Loss関数の値が下がれないので。もしBUGを見つかったら、必ずお伝えいただきます。）
### 3.3 Adv_Loss関数　




このアルゴリズムを色変換の分野に応用する。現在のアルゴリズムでは、ハイライトと白い物体を区別することができず、同様にシャドウと黒い物体を区別することもできない。
このアルゴリズムを使って、これまでのアルゴリズムの欠点を補う。
図１は色変換のOverviewです。
![image](https://user-images.githubusercontent.com/71435435/167250705-28ccd13f-6ecd-427a-aeff-2f55de6be813.png)
