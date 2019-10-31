import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

"""
Bi-linear補間により画像を1.5倍に拡大せよ。

Bi-linear補間とは周辺の４画素に距離に応じた重みをつけることで補完する手法である。 計算量が多いだけ処理時間がかかるが、画質の劣化を抑えることができる。

拡大画像の座標(x', y')を拡大率aで割り、floor(x'/a, y'/a)を求める。
元画像の(x'/a, y'/a)の周囲4画素、I(x,y), I(x+1,y), I(x,y+1), I(x+1, y+1)を求める
それぞれの画素と(x'/a, y'/a)との距離dを求め、重み付けする。 w = d / Sum d
次式によって拡大画像の画素(x',y')を求める。 dx = x'/a - x , dy = y'/a - y

I'(x',y')=(1-dx)(1-dy)*I(x,y)+dx(1-dy)*I(x+1,y)+(1-dx)dy*I(x,y+1)+dxdy*I(x+1,y+1)

"""
#入力
img = cv2.imread("input/imori.jpg")

#処理
#アルゴリズム
# https://daeudaeu.com/programming/c-language/affine/
#affine変換
# http://ipr20.cs.ehime-u.ac.jp/column/gazo_syori/chapter3.html

out = img.copy().astype(np.float32)

def affine(img,a,b,c,d,tx,ty):
    tmp=np.zeros_like(img,dtype=np.float32)
    H,W,C=img.shape
    
    #アフィン行列形式
    K=np.array([[a,b,tx],
                [c,d,ty],
                [0,0,1]])

    #アフィン変換の逆行列
    K_inv=np.linalg.inv(K)

    #各ピクセルずつ動かす
    for i in range(H*d):
        for j in range(W*a):
            for k in range(C):
                #異動後に対応するベクトル
                o=np.array([[i],
                            [j],
                            [1]])
                #逆行列計算 移動前の位置を求める
                k_o=np.dot(K_inv,o)

                #移動前に対応する座標
                x=int(k_o[0])
                y=int(k_o[1])
    
                #はみ出したピクセルは扱わない
                if x<0 or y<0 or x>=W or y>=H:
                    break
                else:
                    tmp[j,i,k]=img[y,x,k]

    #はみ出した画素値を範囲に収める
    tmp=np.where(tmp <0,0,tmp)
    tmp[tmp>255]=255
    tmp=tmp.astype(np.uint8)
    return tmp

out=affine(img,a=1,b=0,c=0,d=1,tx=30,ty=-30)

# #確認b
cv2.imshow("result", out)
cv2.waitKey(0)

#出力
cv2.imwrite("output/q28.jpg", out)