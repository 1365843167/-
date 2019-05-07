# -*- coding:utf-8 -*-
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Lambda, Concatenate, Add, Multiply
#输入64*64 *128   原dense_layer:norm->relu->bottleneck->conv->dropout
def res2net_bottleneck_block(x, f, s=4, expansion=4, use_se_block=False):
    """
    https://github.com/shaoanlu/experiment-with-res2net/blob/master/res2net_block.py
    Arguments:  
        x: input tensor 
        f: number of output  channels
        s: scale dimension  比例尺寸4,即将输入通道数分4组后，每组所含的通道数x/4
    """
    
    num_channels = int(x._keras_shape[-1])
    assert num_channels % s == 0, f"Number of input channel should be a multiple of s. Received nc={num_channels} and s={s}."
    
    input_tensor = x  #num_input_features   64*64 *128
    
    # Conv 1x1
    x = BatchNormalization()(x)  #out:64*64 *128
    x = Activation('relu')(x)  #out:64*64 *128  f
    x = Conv2D(f, 1, kernel_initializer='he_normal', use_bias=False)(x)  #1*1conv,64*64 *128
    #用3x3的组卷积（group convolution）替代原3x3卷积，但未加入SE 单元（block），分为g组则参数量是普通卷积的1/g
    # Conv 3x3
    subset_x = []
    n = f
    w = n // s #s=4,将卷积输入通道数f分为4组，w为每组卷积运算的通道数
    for i in range(s):
        slice_x = Lambda(lambda x: x[..., i*w:(i+1)*w])(x)  #记录分组通道数的编号
        if i > 1:
            slice_x = Add()([slice_x, subset_x[-1]]) #i=0,第一组，直接输入等于输出，没有其它操作
        if i > 0:
            slice_x = BatchNormalization()(slice_x)#i>0,后面3组，都进行3x3的组卷积，由group=1变group=c通道数
            slice_x = Activation('relu')(slice_x)
            slice_x = Conv2D(w, 3, kernel_initializer='he_normal', padding='same', use_bias=False)(slice_x)  #3*3conv,padding='same'即 padding=1 默认的padding='vaild' 即without padding卷积扫特征图时，不够的直接丢弃，而same会进行补0
        subset_x.append(slice_x)  #若Conv 3x3 s=1 p=1 为3个 64*64 *128;若Conv 3x3 s=2 p=1 为3个 32*32 *128 图大小不一致无法拼接
    x = Concatenate()(subset_x) #原始输入64*64 *128 再concat 3个 64*64 *128  增加模型宽度
    #且第二组卷积后输出也与x一起作为第三组的输入，第三组卷积后输出也与x一起作为第四组的输入
    # Conv 1x1
    x = BatchNormalization()(x) #64*64*896 
    x = Activation('relu')(x)
    x = Conv2D(f*expansion, 1, kernel_initializer='he_normal', use_bias=False)(x)
    
    if use_se_block:
        x = se_block(x)
    
    # Add
    if num_channels == f*expansion:
        skip = input_tensor
    else:
        skip = input_tensor
        skip = Conv2D(f*expansion, 1, kernel_initializer='he_normal')(skip)
    out = Add()([x, skip])
    return out

def se_block(input_tensor, c=16):
    num_channels = int(input_tensor._keras_shape[-1]) # Tensorflow backend
    bottleneck = int(num_channels // c)
 
    se_branch = GlobalAveragePooling2D()(input_tensor)
    se_branch = Dense(bottleneck, use_bias=False, activation='relu')(se_branch)
    se_branch = Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)
    
    out = Multiply()([input_tensor, se_branch]) 
    return out