import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

#入力
img = cv2.imread("input/imori.jpg")
# img = cv2.imread("araineru.png")
# img = func.RGB2GRAY(img)

#処理
#アルゴリズム
# https://daeudaeu.com/programming/c-language/affine/
#affine変換
# http://ipr20.cs.ehime-u.ac.jp/column/gazo_syori/chapter3.html

# out = img.copy().astype(np.float32)


def affine(img,a,b,c,d,tx,ty):
    H,W,C=img.shape
    hh=np.ceil(H*d).astype(np.int)
    ww=np.ceil(W*a).astype(np.int)
    cc=int(C)
    
    tmp=np.zeros((hh, ww, cc))

    #アフィン変換　行列形式
    K=np.array([[a,b,tx],
                [c,d,ty],
                [0,0,1]])

    #アフィン変換の逆行列
    K_inv=np.linalg.inv(K)

    #各ピクセルずつ動かす
    for i in range(ww):  # 変換後の座標
        for j in range(hh):
            for k in range(cc):#ここにCぶち込んでもダメだった、intに直さないと
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
                if x < 0 or y < 0 or x >= ww or y >= hh:
                    break
                # elif x>=ww or y>=hh:
                #     break
                else:
                    tmp[j,i,k]=img[y,x,k]

    #はみ出した画素値を範囲に収める
    tmp=np.where(tmp <0,0,tmp)
    tmp[tmp>255]=255
    tmp=tmp.astype(np.uint8)
    return tmp

# out=affine(img,a=1,b=0,c=0,d=1,tx=30,ty=-30)
out=affine(img,a=1.3,b=0,c=0,d=0.8,tx=0,ty=0)

out=affine(out,a=1,b=0,c=0,d=1,tx=30,ty=-30)

# #確認b
cv2.imshow("result", out)
cv2.waitKey(0)

#出力
cv2.imwrite("output/q29.jpg", out)