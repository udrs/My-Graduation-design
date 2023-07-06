URL of my Graduation Thesis : https://github.com/wenyihan4396/My-Graduation-design/blob/main/%E5%8D%92%E6%A5%AD%E8%AB%96%E6%96%87.pdf
# BRDF-Network for recoloring
# 
## 1.材質を予測するAIモデル
## We first implement the paper : 
## Highlight-Aware Two-Stream Network for Single-Image SVBRDF Acquisition
リプロダクションの過程は三つに分かれる。

### １．DataSetの処理
### ２．Network　Modelの構築
#### 2.1 STとHAのTwo-Stream　Network
#### 2.2 Outputは四つなので、2.1に合弁されたfeature　mapは四つのFU　Networkに分かれた。
#### 2.3 予測値の精度を上がるために、perceptual adversarial lossを使って、 two global context discriminators　and two local context discriminatorsをトレーニングする。
### ３．Ground　TruthとLableに基づいてのLoss関数
#### 3.1 SVBRDF_Loss関数 
#### 3.2 Rendering_Loss関数
### 3.3 Adv_Loss関数　

## 2.Recoloring

このアルゴリズムを色変換の分野に応用する。現在のアルゴリズムでは、ハイライトと白い物体を区別することができず、同様にシャドウと黒い物体を区別することもできない。
このアルゴリズムを使って、これまでのアルゴリズムの欠点を補う。
図１は色変換のOverviewです。
![image](https://user-images.githubusercontent.com/71435435/167250705-28ccd13f-6ecd-427a-aeff-2f55de6be813.png)

Evironment:
CUDA 10.4と11.4、対応するpytorchのバージョンはpytorchのホームページで確認できます。
Python3.6

## 3.comparision
origin color based rendering
![image](https://github.com/wenyihan4396/Two-Stream-BRDF-Network_Reproduction/blob/main/origin_color.gif)

recolored based rendering
![image](https://github.com/wenyihan4396/Two-Stream-BRDF-Network_Reproduction/blob/main/color_changed.gif)

## 4.Recoloring comparision
![image](https://github.com/wenyihan4396/Two-Stream-BRDF-Network_Reproduction/blob/main/recoloring%20algorithm%20comparision.png)
