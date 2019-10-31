import numpy as np


def BGR2RGB(img):
    b=img[:,:,0].copy()#img[行:列:色]
    g=img[:,:,1].copy()
    r=img[:,:,2].copy()

    #BGR -> RGB
    img[:,:,0]=r
    img[:,:,1]=g
    img[:,:,2]=b

    return img


def RGB2GRAY(img):
    #与えられた画像を数値値    
    img=img.astype(np.float32)
    b=img[:,:,0].copy()#img[行:列:色]
    g=img[:,:,1].copy()
    r=img[:,:,2].copy()

    img= 0.2126*r + 0.7152*g + 0.0722*b

    #画像を複合して戻す、符号なしの整数にする
    return img.astype(np.uint8)


def binarization(img,th=128):
    img=img.astype(np.float)
    img[img<th]=0
    img[img>=th]=255
    return img          

def Obinarization(img):
    img=img.astype(np.float)
    max_sigma=0
    max_t=0

    for _t in range(1,255):
        lis0=img[np.where(img < _t)]#クラス0
        m0=lis0.mean() if len(lis0) > 0 else 0 #クラス0の平均値
        w0=len(lis0) / len(img)

        lis1=img[np.where(img >= _t)]
        m1=lis1.mean() if len(lis1) > 0 else 0
        w1=len(lis1) / len(img)

        sigma=w0*w1*((m0-m1)**2)

        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # print("threshold:",max_t)
    img[img < max_t]=0
    img[img >=max_t]=255

    return img

# BGR -> HSV
def BGR2HSV(_img):
    img=_img.astype(np.float)
    img = img.copy() / 255.

    hsv = np.zeros_like(img, dtype=np.float32)

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)

    # H
    hsv[..., 0][np.where(max_v == min_v)]= 0
    ## if min == B
    ind = np.where(min_arg == 0)
    hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
    ## if min == R
    ind = np.where(min_arg == 2)
    hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
    ## if min == G
    ind = np.where(min_arg == 1)
    hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
        
    # S
    hsv[..., 1] = max_v.copy() - min_v.copy()

    # V
    hsv[..., 2] = max_v.copy()
    
    return hsv


def HSV2BGR(_img, hsv):
    img=_img.astype(np.float)
    img = img.copy() / 255.

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()

    out = np.zeros_like(img)

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    C = S
    H_ = H / 60.
    X = C * (1 - np.abs( H_ % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i+1)))
        out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
        out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
        out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

    out[np.where(max_v == min_v)] = 0
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    return out


def tone_reduction(img):
    img2=img.copy().astype(np.float32)
    #img[:,:,色]の各値を量しかする
    img2=img2//64*64+32
    return img2.astype(np.uint8)

def average_pooling(img,gx=8,gy=8):
    #事前にgxとgyは画像サイズを確認して入力する必要あり
    out=img.copy().astype(np.float32)
    
    for i in range(int(out.shape[0]/gx)):
        for j in range(int(img.shape[1]/gy)):
            for k in range(img.shape[2]):
                out[gx*i:gx*(i+1),gy*j:gy*(j+1),k]=np.mean(out[gx*i:gx*(i+1),gy*j:gy*(j+1),k])
                
    return out.astype(np.uint8)

def max_pooling(img,gx=8,gy=8):
    #事前にgxとgyは画像サイズを確認して入力する必要あり
    out=img.copy().astype(np.float32)
    
    for i in range(int(out.shape[0]/gx)):
        for j in range(int(img.shape[1]/gy)):
            for k in range(img.shape[2]):
                out[gx*i:gx*(i+1),gy*j:gy*(j+1),k]=np.max(out[gx*i:gx*(i+1),gy*j:gy*(j+1),k])
                
    return out.astype(np.uint8)


def gaussian(x,y,sigma=1.3):
    const=1./np.sqrt(2.*np.pi)*sigma
    expn=np.exp(-(x**2+y**2)/(2.*sigma**2))
    return const*expn

def gaussian_filter(img,size=3,sigma=1.3):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:#https://teratail.com/questions/146318
        img=np.expand_dims(img,axis=-1)#imgの次元が3に満たないことを想定、縦、横の後に次元を追加
        H,W,C=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2,C),dtype=np.float32)#imgより0パでイング文大きい配列用意
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)#格納

    #gauss 注目画像をずらしながらその周囲にフィルタをかける（フィルタサイズで抽出した画像とフィルタのアダマール積）
    K = np.zeros((size, size), dtype=np.float32)
    for x in range(-zero_pad, -zero_pad + size):
        for y in range(-zero_pad, -zero_pad + size):
            K[y + zero_pad, x + zero_pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()

    tmp=out.copy()

    #フィルタリング
    for i in range(W):
        for j in range(H):
            for c in range(C):
                out[zero_pad+i,zero_pad+j,c]=np.sum(K*tmp[i:i+size,j:j+size,c])

    #http://optie.hatenablog.com/entry/2018/03/21/185647#32-%E3%82%AC%E3%82%A6%E3%82%B7%E3%82%A2%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF
    # https://note.nkmk.me/python-numpy-clip/
    out=np.clip(out,0,255)
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)

    return out


def median_filter(img,size=3,sigma=1.3):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:#https://teratail.com/questions/146318
        img=np.expand_dims(img,axis=-1)#imgの次元が3に満たないことを想定、縦、横の後に次元を追加
        H,W,C=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2,C),dtype=np.float32)#imgより0パでイング文大きい配列用意
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)#格納

    #フィルタを作成する必要はないので、n.median()で中央値取得すればよい 

    tmp=out.copy()

    #フィルタリング
    for i in range(W):
        for j in range(H):
            for c in range(C):
                out[zero_pad+i,zero_pad+j,c]=np.median(tmp[i:i+size,j:j+size,c])


    #http://optie.hatenablog.com/entry/2018/03/21/185647#32-%E3%82%AC%E3%82%A6%E3%82%B7%E3%82%A2%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF
    # https://note.nkmk.me/python-numpy-clip/
    out=np.clip(out,0,255)
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)

    return out


def mean_filter(img,size=3):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:#https://teratail.com/questions/146318
        img=np.expand_dims(img,axis=-1)#imgの次元が3に満たないことを想定、縦、横の後に次元を追加
        H,W,C=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2,C),dtype=np.float32)#imgより0パでイング文大きい配列用意
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)#格納

    #gauss 注目画像をずらしながらその周囲にフィルタをかける（フィルタサイズで抽出した画像とフィルタのアダマール積）
    K = np.ones((size, size), dtype=np.float32)
    K = K/K.sum()
    # https://algorithm.joho.info/image-processing/average-filter-smooth/

    tmp=out.copy()

    #フィルタリング
    for i in range(W):
        for j in range(H):
            for c in range(C):
                out[zero_pad+i,zero_pad+j,c]=np.sum(K*tmp[i:i+size,j:j+size,c])

    #http://optie.hatenablog.com/entry/2018/03/21/185647#32-%E3%82%AC%E3%82%A6%E3%82%B7%E3%82%A2%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF
    # https://note.nkmk.me/python-numpy-clip/
    out=np.clip(out,0,255)
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)

    return out

def LoG_filter(img,size=5,sigma=3):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:#https://teratail.com/questions/146318
        img=np.expand_dims(img,axis=-1)#imgの次元が3に満たないことを想定、縦、横の後に次元を追加
        H,W,C=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2,C),dtype=np.float32)#imgより0パでイング文大きい配列用意
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)#格納

    #  注目画像をずらしながらその周囲にフィルタをかける（フィルタサイズで抽出した画像とフィルタのアダマール積）
    K = np.zeros((size, size), dtype=np.float32)
    for x in range(-zero_pad, -zero_pad + size):
        for y in range(-zero_pad, -zero_pad + size):
            const=x**2+y**2-sigma**2
            K[y + zero_pad, x + zero_pad] = const*np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma**6)
    K /= K.sum()

    tmp=out.copy()

    #フィルタリング
    for i in range(W):
        for j in range(H):
            for c in range(C):
                out[zero_pad+i,zero_pad+j,c]=np.sum(K*tmp[i:i+size,j:j+size,c])

    #http://optie.hatenablog.com/entry/2018/03/21/185647#32-%E3%82%AC%E3%82%A6%E3%82%B7%E3%82%A2%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF
    # https://note.nkmk.me/python-numpy-clip/
    out=np.clip(out,0,255)
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)

    return out

def motion_filter(img,size=3):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2,C),dtype=np.float32)
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)

    #filter 注目画像をずらしながらその周囲にフィルタをかける（フィルタサイズで抽出した画像とフィルタのアダマール積）
    K = np.ones((size, size), dtype=np.float32)
    for i in range(K.shape[1]):
        for j in range(K.shape[0]):
            if i==j:
                K[j,i]=K[j,i]/np.diag(K).sum()
            else:
                K[j,i]=0
    
    tmp=out.copy()

    #フィルタリング
    for i in range(W):
        for j in range(H):
            for c in range(C):
                out[zero_pad+i,zero_pad+j,c]=np.sum(K*tmp[i:i+size,j:j+size,c])

    out=np.clip(out,0,255)
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)
    return out

def max_min_filter(img,size=3):
    img=img.astype(np.float32)
    H,W=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2),dtype=np.float32)
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)
    
    tmp=out.copy()

    #フィルタリング
    for y in range(H):
        for x in range(W):
            out[zero_pad + y, zero_pad + x] = np.max(tmp[y: y + size, x: x + size]) - np.min(tmp[y: y + size, x: x + size])
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)
    return out

def differential_filter(img,size=3,axis=0):
    img=img.astype(np.float32)
    H,W=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2),dtype=np.float32)
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)
    
    #kernel
    if axis==0:#行方向
        K=[ [0,0,0],[-1,1,0],[0,0,0]]
    else:#列方向
        K=[ [0,-1,0],[0,1,0],[0,0,0]]

    tmp=out.copy()

    #フィルタリング
    for y in range(H):
        for x in range(W):#sumで集約する必要がある
            out[zero_pad + y, zero_pad + x] = np.sum(K*tmp[y: y + size, x: x + size])
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)
    return out

def sobel_filter(img,size=3,axis=0):
    img=img.astype(np.float32)
    H,W=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2),dtype=np.float32)
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)
    
    #kernel
    if axis==0:#行方向
        K=[ [1,2,1],[0,0,0],[-1,-2,-1]]
    else:#列方向
        K=[ [1,0,-1],[2,0,-2],[1,0,-1]]

    tmp=out.copy()

    #フィルタリング
    for y in range(H):
        for x in range(W):#sumで集約する必要がある
            out[zero_pad + y, zero_pad + x] = np.sum(K*tmp[y: y + size, x: x + size])
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)
    return out

def Prewitt_filter(img,size=3,axis=0):
    img=img.astype(np.float32)
    H,W=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2),dtype=np.float32)
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)
    
    #kernel
    if axis==0:#行方向
        K=[ [-1,0,1],[-1,0,1],[-1,0,1]]
    else:#列方向
        K=[ [-1,-1,-1],[0,0,0],[1,1,1]]

    tmp=out.copy()

    #フィルタリング
    for y in range(H):
        for x in range(W):#K書けただけだとサイズが異なるからsumで集約する必要がある
            out[zero_pad + y, zero_pad + x] = np.sum(K*tmp[y: y + size, x: x + size])
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)
    return out

def Laplacian_filter(img,size=3):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape

    #0パディング
    zero_pad=size//2 #余る=奇数サイズ
    out=np.zeros((H+zero_pad*2,W+zero_pad*2,C),dtype=np.float32)
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)

    K=[ [0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]

    tmp=out.copy()

    #フィルタリング
    for i in range(W):
        for j in range(H):
            for c in range(C):
                out[zero_pad+i,zero_pad+j,c]=np.sum(K*tmp[i:i+size,j:j+size,c])

    out=np.clip(out,0,255)
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)

    return out

def Emboss_filter(img,size=3):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape

    #0パディング
    zero_pad=size//2 
    out=np.zeros((H+zero_pad*2,W+zero_pad*2,C),dtype=np.float32)
    out[zero_pad:H+zero_pad,zero_pad:W+zero_pad]=img.copy().astype(np.float32)

    K=[ [-2,-1,0],[-1,1,1],[0,1,2]]

    tmp=out.copy()

    #フィルタリング
    for i in range(W):
        for j in range(H):
                out[zero_pad+j,zero_pad+i]=np.sum(K*tmp[j:j+size,i:i+size])-128

    out=np.clip(out,0,255)
    out=out[zero_pad:zero_pad+H,zero_pad:zero_pad+W].astype(np.uint8)

    return out

def hist_normalization(img,a=0,b=255):
    img=img.astype(np.float32)
    out=img.copy()

    c=np.min(out)
    d=np.max(out)

    const = (b-a)/(d-c)
    out = const*(out-c)+a
    out=np.where(out<a,a,out)
    out=np.where(out>b,b,out)
    out = out.astype(np.uint8)
    return out

def hist_manipulate(img,m0,s0):
    img=img.astype(np.float32)
    out=img.copy()
    #平均m,標準偏差s
    m=np.mean(out)
    s=np.std(out)

    #manipulate 平坦化処理
    out=(s0/s)*(out-m)+m0
    out=np.where(out<0,0,out)
    out=np.where(out>255,255,out)
    out = out.astype(np.uint8)
    return out


def hist_equalization(img,z_max=255):
    out=img.copy().astype(np.float32)
    H,W,C=out.shape
    S=H*W*C #画素数の総数は画像の総ピクセルx色

    sum_=0.#度数の和
    # Z_max=255 #画素値の最大値 0~255なので255

    for i in range(1,255):
        ind= np.where(out==i)#画素値の要素列を抽出
        sum_= sum_ + len(out[ind])#濃度の度数を足す
        const = z_max/S
        out[ind]=const * sum_

    return out.astype(np.uint8)


def Gamma_correction(img,c=1.,gamma=2.2):
    out=img.copy().astype(np.float32)
    out /= 255.#正規化
    out=(1/c * out)**(1./gamma)
    out *=255.#元の幅に拡張
    out=out.astype(np.uint8)
    return out


def nn_interpolate(img,alpha=1.):
    out=img.copy().astype(np.float32)
    H,W,C=img.shape

    aH=int(alpha*H)
    aW=int(alpha*W)

    # https://note.nkmk.me/python-numpy-reshape-usage/
    y=np.arange(aH).repeat(aW).reshape(aW,-1)#aW行並ぶようにreshape自動行う
    y=np.round(y / alpha).astype(np.int)
    #https://note.nkmk.me/python-numpy-tile/
    x=np.tile(np.arange(aW),(aH,1))
    x=np.round(x / alpha).astype(np.int)

    out=img[y,x]
    out=out.astype(np.uint8)
    return out


def bl_interpolate(img, ax=1., ay=1.):
	H, W, C = img.shape

	aH = int(ay * H)
	aW = int(ax * W)

	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))

	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-2)
	iy = np.minimum(iy, H-2)

	dx = x - ix
	dy = y - iy

	dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
	dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# Bi-cubic interpolation
def bc_interpolate(img, ax=1., ay=1.):
	H, W, C = img.shape

	aH = int(ay * H)
	aW = int(ax * W)

	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))
	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-1)
	iy = np.minimum(iy, H-1)

	dx2 = x - ix
	dy2 = y - iy
	dx1 = dx2 + 1
	dy1 = dy2 + 1
	dx3 = 1 - dx2
	dy3 = 1 - dy2
	dx4 = 1 + dx3
	dy4 = 1 + dy3

	dxs = [dx1, dx2, dx3, dx4]
	dys = [dy1, dy2, dy3, dy4]

    #重み
	def weight(t):
		a = -1.
		at = np.abs(t)
		w = np.zeros_like(t)
		ind = np.where(at <= 1)
		w[ind] = ((a+2) * np.power(at, 3) - (a+3) * np.power(at, 2) + 1)[ind]
		ind = np.where((at > 1) & (at <= 2))
		w[ind] = (a*np.power(at, 3) - 5*a*np.power(at, 2) + 8*a*at - 4*a)[ind]
		return w

	w_sum = np.zeros((aH, aW, C), dtype=np.float32)
	out = np.zeros((aH, aW, C), dtype=np.float32)

	for j in range(-1, 3):
		for i in range(-1, 3):
			ind_x = np.minimum(np.maximum(ix + i, 0), W-1)
			ind_y = np.minimum(np.maximum(iy + j, 0), H-1)

			wx = weight(dxs[i+1])
			wy = weight(dys[j+1])
			wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
			wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)

			w_sum += wx * wy
			out += wx * wy * img[ind_y, ind_x]

	out /= w_sum
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


def affine(img, a, b, c, d, tx, ty):
    H, W, C = img.shape

    img = np.zeros((H+2, W+2, C), dtype=np.float32)
    img[1:H+1, 1:W+1] = _img

    H_new = np.round(H * d).astype(np.int)
    W_new = np.round(W * a).astype(np.int)
    out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    out[y_new, x_new] = img[y, x]

    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)

    return out