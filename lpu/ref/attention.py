import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, mask=None):
    seq_len, hidden_dim = x.shape[0], x.shape[1]
    q = np.matmul(x, w_q)  # shape (seq_len, hidden_dim)
    k = np.matmul(x, w_k)  # shape (seq_len, hidden_dim)
    v = np.matmul(x, w_v)  # shape (seq_len, hidden_dim)

    head_dim = hidden_dim // num_heads
    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2) # shape (num_heads, seq_len, head_dim)
    k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2) # shape (num_heads, seq_len, head_dim)
    v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2) # shape (num_heads, seq_len, head_dim)

    k_transpose = k.transpose(0, 2, 1)  # shape (num_heads, head_dim, seq_len)
    attn_scores = np.matmul(q, k_transpose) / np.sqrt(head_dim)  # shape (num_heads, seq_len, seq_len)
    if mask is not None:
        attn_scores = np.where(mask == 0, -1e9, attn_scores)

    attn_weights = softmax(attn_scores, axis=-1)  # shape (num_heads, seq_len, seq_len)
    out = np.matmul(attn_weights, v)  # shape (num_heads, seq_len, head_dim)
    out = out.transpose(1, 0, 2).reshape(seq_len, hidden_dim)
    output = np.matmul(out, w_o)  # shape (seq_len, hidden_dim)
    return output

def multi_head_attention_tiled_1(x, w_q, w_k, w_v, w_o, num_heads):
    # Reshape weight matrices to avoid reshaping q, k, v later
    seq_len, hidden_dim = x.shape[0], x.shape[1]
    head_dim = hidden_dim // num_heads
    w_q_reshaped = w_q.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)  # shape (num_heads, hidden_dim, head_dim)
    w_k_reshaped = w_k.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)  # shape (num_heads, hidden_dim, head_dim)
    w_v_reshaped = w_v.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)  # shape (num_heads, hidden_dim, head_dim)
    w_o_reshaped = w_o.reshape(num_heads, head_dim, hidden_dim)

    pv = np.zeros((num_heads, seq_len, head_dim))
    output = np.zeros((seq_len, hidden_dim))
    for h in range(num_heads):
        q = np.matmul(x, w_q_reshaped[h])  # shape (seq_len, head_dim)
        k = np.matmul(x, w_k_reshaped[h])  # shape (seq_len, head_dim)
        v = np.matmul(x, w_v_reshaped[h])  # shape (seq_len, head_dim)
        k_transpose = k.transpose()  # shape (head_dim, seq_len)
        attn_scores = np.matmul(q, k_transpose) / np.sqrt(head_dim)  # shape (seq_len, seq_len)
        attn_weights = softmax(attn_scores, axis=-1) # shape (seq_len, seq_len)
        pv[h] = np.matmul(attn_weights, v) # shape (seq_len, head_dim)
        output += np.matmul(pv[h], w_o_reshaped[h]) # shape (seq_len, hidden_dim)

    return output

def multi_head_attention_tiled_2(x, w_q, w_k, w_v, w_o, num_heads):
    # Reshape weight matrices to avoid reshaping q, k, v later
    seq_len, hidden_dim = x.shape[0], x.shape[1]
    head_dim = hidden_dim // num_heads
    w_q_reshaped = w_q.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)  # shape (num_heads, hidden_dim, head_dim)
    w_k_reshaped = w_k.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)  # shape (num_heads, hidden_dim, head_dim)
    w_v_reshaped = w_v.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)  # shape (num_heads, hidden_dim, head_dim)

    pv = np.zeros((num_heads, seq_len, head_dim))
    for h in range(num_heads):
        q = np.matmul(x, w_q_reshaped[h])  # shape (seq_len, head_dim)
        k = np.matmul(x, w_k_reshaped[h])  # shape (seq_len, head_dim)
        v = np.matmul(x, w_v_reshaped[h])  # shape (seq_len, head_dim)
        k_transpose = k.transpose()  # shape (head_dim, seq_len)
        attn_scores = np.matmul(q, k_transpose) / np.sqrt(head_dim)  # shape (seq_len, seq_len)
        attn_weights = softmax(attn_scores, axis=-1)  # shape (seq_len, seq_len)
        pv[h] = np.matmul(attn_weights, v)

    pv = pv.transpose(1, 0, 2).reshape(seq_len, hidden_dim)  # shape (seq_len, hidden_dim)
    output = np.matmul(pv, w_o)  # shape (seq_len, hidden_dim)
    return output

def flash_attention_3(x, w_q, w_k, w_v, w_o, num_heads):
    # Reshape weight matrices to avoid reshaping q, k, v later
    seq_len, hidden_dim = x.shape[0], x.shape[1]
    seq_tile = 16
    head_dim = hidden_dim // num_heads

    w_q_reshaped = w_q.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)
    w_k_reshaped = w_k.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)
    w_v_reshaped = w_v.reshape(hidden_dim, num_heads, head_dim).transpose(1, 0, 2)

    O = np.zeros((num_heads, seq_len, head_dim))
    l = np.zeros((num_heads, seq_len, 1))
    m = np.full((num_heads, seq_len, 1), float('-inf'))

    for h in range(num_heads):
        for q_start in range(0, seq_len, seq_tile):
            q_end = min(q_start + seq_tile, seq_len)
            x_q_tile = x[q_start:q_end]
            q_tile = np.matmul(x_q_tile, w_q_reshaped[h])  # shape (seq_tile, head_dim)
            m_tile = m[h, q_start:q_end]
            l_tile = l[h, q_start:q_end]
            for kv_start in range(0, seq_len, seq_tile):
                kv_end = min(kv_start + seq_tile, seq_len)
                x_kv_tile = x[kv_start:kv_end]
                k_tile = np.matmul(x_kv_tile, w_k_reshaped[h])  # shape (seq_tile, head_dim)
                v_tile = np.matmul(x_kv_tile, w_v_reshaped[h])  # shape (seq_tile, head_dim)
                qk = np.matmul(q_tile, k_tile.transpose()) / np.sqrt(head_dim)  # shape (num_heads, seq_tile, seq_tile)

                m_prev = m_tile
                m_curr = np.maximum(m_prev, np.max(qk, axis=-1, keepdims=True))
                exp_scores = np.exp(qk - m_curr) # shape (num_heads, seq_tile, seq_tile)
                l_prev = l_tile
                exp_m_diff = np.exp(m_prev - m_curr)

                l_curr = exp_m_diff * l_prev + np.sum(exp_scores, axis=-1, keepdims=True)

                O[h, q_start:q_end] = (exp_m_diff * l_prev / l_curr) * O[h, q_start:q_end] + np.matmul(exp_scores, v_tile) / l_curr

                m[h, q_start:q_end] = m_curr
                l[h, q_start:q_end] = l_curr

    O = O.transpose(1, 0, 2).reshape(seq_len, hidden_dim)
    output = np.matmul(O, w_o)  # shape (seq_len, hidden_dim)
    return output

seq_len = 10
hidden_dim = 64
num_heads = 16

x = np.random.rand(seq_len, hidden_dim)
w_q = np.random.rand(hidden_dim, hidden_dim)
w_k = np.random.rand(hidden_dim, hidden_dim)
w_v = np.random.rand(hidden_dim, hidden_dim)
w_o = np.random.rand(hidden_dim, hidden_dim)

output_golden = multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads)
output_flash_attention_3 = flash_attention_3(x, w_q, w_k, w_v, w_o, num_heads)
assert np.allclose(output_golden, output_flash_attention_3), "Outputs do not match!"
print("output_flash_attention_3 match successfully!")
