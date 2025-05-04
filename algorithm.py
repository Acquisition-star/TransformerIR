# Algorithm 1 SCA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 池化卷积 input (C, H, W) -> (C, 1)
A = W_pool(input)
# 注意力计算
output = input * A


# Algorithm 2 MDTA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 卷积映射 Q, K, V -> (C, H, W)
query, key, value = Wd(Wp(input)).chunk(3, dim=0)
# 向量重塑 Q, K, V -> (C, HW)
query, key, value = rearrange(query, key, value)
# 注意力计算
A = mm(query, key.transpose(-2, -1)) * alpha
A = softmax(A, dim=-1)
output = mm(A, value)
# 复原 output -> (C, H, W)
output = rearrange(output)


# Algorithm 3 SGA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 卷积映射 Q, K, V -> (C, H, W)
query, key, value = Wd(Wp(input)).chunk(3, dim=0)
# 向量重塑 Q, K, V -> (C, HW)
query, key, value = rearrange(query, key, value)
# 归一化处理
query = normalize(query, dim=-1)
key = normalize(key, dim=-1)
# 注意力计算
A = mm(query, key.transpose(-2, -1)) * temperature
A = ReLU(A, dim=-1)
output = mm(A, value)
# 复原 output -> (C, H, W)
output = rearrange(output)


# Algorithm 4 WA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 窗口划分 input (C, H, W) -> (N, C, Wh, Ww)
windows = partition(input, windows_size)
# 线性映射 Q, K, V
q, k, v = linear_qkv(windows).chunk(3, dim=-1)
# 多头注意力重塑 Q, K, V -> (N, heads, Wh*Ww, C//heads)
q, k, v = rearrange(q, k, v)
# 注意力计算
A = mm(q * scale, k.transpose(-2, -1))
A = softmax(A, dim=-1)
# 相对位置偏移
A = A + relative_position_bias
output = mm(A, v)
# 复原 output -> (C, H, W)
output = rearrange(output)


# Algorithm 5 SWA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 像素位移
x = roll(input, shifts)
# 窗口划分 input (C, H, W) -> (N, C, Wh, Ww)
windows = partition(x, windows_size)
# 线性映射 Q, K, V
q, k, v = linear_qkv(windows).chunk(3, dim=-1)
# 多头注意力重塑 Q, K, V -> (N, heads, Wh*Ww, C//heads)
q, k, v = rearrange(q, k, v)
# 注意力计算
A = mm(q * scale, k.transpose(-2, -1))
A = softmax(A, dim=-1)
# 相对位置偏移
A = A + relative_position_bias
output = mm(A, v)
# 复原 output -> (C, H, W)
output = unroll(output, shifts)
output = rearrange(output)


# Algorithm 6 SPA
# Input: features (C, H, W)
# Output: features (C, H, W)

# 条内注意力 特征提取 Xh -> (H, W, C//2) Xv -> (W, H, C//2)
Xh, Xv = conv(input).chunk(2, dim=0)
Xh = rearrange(Xh)
Xv = rearrange(Xv)
# 线性映射 Q, K, V
q_h, k_h, v_h = linear_qkv(Xh).chunk(3, dim=1)
q_v, k_v, v_v = linear_qkv(Xv).chunk(3, dim=1)
# 注意力计算
A_h = mm(q_h, k_h.transpose(-2, -1)) * sqrt(d)
A_v = mm(q_v, k_v.transpose(-2, -1)) * sqrt(d)
A_h = softmax(A_h, dim=-1) * v_h
A_v = softmax(A_v, dim=-1) * v_v
# 注意力融合
output = conv(cat(A_h, A_v))

# 条间注意力 特征提取 X_h, X_v -> (C//2, H, W)
Xh, Xv = conv(input).chunk(2, dim=0)
# 卷积映射 Q, K, V
q_h, k_h, v_h = conv_qkv(Xh).chunk(3, dim=0)
q_v, k_v, v_v = conv_qkv(Xv).chunk(3, dim=0)
# 向量重塑 x_h -> (H, W * C//2) x_v -> (W, H * C//2)
q_h, k_h, v_h = rearrange(q_h, k_h, v_h)
q_v, k_v, v_v = rearrange(q_v, k_v, v_v)
# 注意力计算
A_h = mm(q_h, k_h.transpose(-2, -1)) * sqrt(d)
A_v = mm(q_v, k_v.transpose(-2, -1)) * sqrt(d)
A_h = softmax(A_h, dim=-1) * v_h
A_v = softmax(A_v, dim=-1) * v_v
# 向量复原 x_h -> (C//2, H, W) x_v -> (C//2, W, H)
q_h, k_h, v_h = rearrange(q_h, k_h, v_h)
q_v, k_v, v_v = rearrange(q_v, k_v, v_v)
# 注意力融合
output = conv(cat(A_h, A_v))


# Algorithm 7 EA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 卷积映射 Q, K, V -> (C, HW)
query = rearrange(conv_q(input))
key = rearrange(conv_k(input))
value = rearrange(conv_v(input))
# 行归一化
query = softmax(query, dim=1)
# 列归一化
key = softmax(key, dim=0)
# 注意力计算
context = mm(key.transpose(-2, -1), value)
output = mm(query, context)
# 复原 output -> (C, H, W)
output = rearrange(output)


# Algorithm 8 TMSA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 卷积映射 Q, K, V -> (head, HW, C//head)
Q, K, V = rearrange(conv_qkv(input))
Q1 = normalize(Q, dim=-1)
K1 = normalize(K, dim=-1)
Qh = normalize(ReLU(Q)**factor, dim=-1)
Kh = normalize(ReLU(K)**factor, dim=-1)
# 注意力计算
Q_K_V1 = mm(Q1, mm(K1.transpose(-2, -1), V))
Q_K_Vh = mm(Qh, mm(Kh.transpose(-2, -1), V))
# 余项展开计算计算
Ones_Vh = sum(V, dim=-2).unsqueeze(1)
K_Ones_1 = sum(K.transpose(-2, -1), dim=-2).unsqueeze(1)
Q_K_Ones_1 = mm(Q, K_Ones_1)
K_Ones_h = sum(Kh.transpose(-2, -1), dim=-2).unsqueeze(1)
Q_K_Ones_h = mm(Q, K_Ones_h)
N = Ones_Vh + Q_K_V1 + Q_K_Ones_h
D = H * W + Q_K_Ones_1 + Q_K_Ones_h + 1e-6
# 复原 output -> (C, H, W)
output = rearrange(div(N, D)) + CPE(V)
output = conv(output)


# Algorithm 9 FSAS
# Input: features (C, H, W)
# Output: features (C, H, W)
# 卷积映射 Q, K, V
Q, K, V = conv_qkv(input).chunk(3, dim=0)
# 向量重塑 Q, K -> (C, H//patch, W//patch, patch, patch)
Q = rearrange(Q)
K = rearrange(K)
# 傅里叶变换计算
Q_fft = rfft2(Q)
K_fft = rfft2(K)
A_fft = Q_fft * K_fft
A = irfft(A_fft)
# 向量重塑 A -> (C, H, W)
A = rearrange(A)
output = conv(V * A)


# Algorithm 10 BRA
# Input: features (C, H, W)
# Output: features (C, H, W)
# 划分 input (C, H, W) -> (S^2, HW/S^2, C)
x = patchify(input, patch_size=(H // S, W // S))
# 线性映射 Q, K, V
query, key, value = linear_qkv(x).chunk(3, dim=-1)
# 区域压缩 Q, K -> (S^2, C)
query_r, key_r = query.mean(dim=1), key.mean(dim=1)
# 区域相关性图
A_r = mm(query_r, key_r.transpose(-1, -2))
# 计算路由索引矩阵
I_r = topk(A_r, k).index
# 收集键值对
key_g = gather(key, I_r)
value_g = gather(value, I_r)
# 注意力计算
A = mm(query, key_g.transpose(-2, -1))
A = softmax(A, dim=-1)
output = mm(A, value_g) + lce(value)
# 复原 output -> (C, H, W)
output = unpatchify(output, patch_size=(H // S, W // S))
