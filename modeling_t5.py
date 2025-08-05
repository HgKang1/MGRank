def vanila_gaussian_weight(key_length, batch_size, t5_cand_pos_s, t5_cand_pos_e, layer_idx, initial_std, std_scaling_factor, device):
            x = torch.arange(key_length, device=device).repeat(batch_size, 1).float()
            sigma = initial_std + layer_idx * std_scaling_factor
            t5_cand_pos_s = t5_cand_pos_s.view(batch_size, 1).to(device)
            t5_cand_pos_e = t5_cand_pos_e.view(batch_size, 1).to(device)
            
            t5_center = torch.round((t5_cand_pos_s + t5_cand_pos_e) / 2)
            weights = torch.exp(-(((x - t5_center) / 50) ** 2) / (2 * sigma**2))
            # weights = weights / weights.max(dim=1, keepdim=True)[0]  # normalize to [0,1]
            
            mask_in_range = (x >= t5_cand_pos_s) & (x <= t5_cand_pos_e)  # mask for s~e
            mask_in_range = mask_in_range.view(batch_size, 1, 1, key_length)
            weights = weights.view(batch_size, 1, 1, key_length)
            weights[mask_in_range] = 1.0
            
            mask = (t5_cand_pos_s >= 512).view(batch_size, 1, 1, 1)
            weights = torch.where(mask, torch.ones_like(weights), weights)
            
            return weights


def gaussian_mixture_weighting(
    scores,  
    underscore_mask,
    topk,
    gaussian_weight,   
    sim_word_idx_tensor,
):
    device = scores.device
    batch_size, num_heads, seq_length, key_length = scores.shape

    lambda_weights = [0.1,0.1,0.1]
    std = [0.15,0.1,0.05]
    x = torch.arange(key_length, device=device).repeat(batch_size, 1).float()
    mix_gaussian = torch.zeros((batch_size, key_length), device=device)
    
    i=0
    for k_idx, weight in enumerate(lambda_weights[:topk]):
        center = sim_word_idx_tensor[:, :, k_idx].squeeze(1).squeeze(1)
        center =center[:, 0].unsqueeze(1)
        gaussian_k = torch.exp(-(((x - center)/50) ** 2) / (2 * (std[i] ** 2)))
        # gaussian_k = torch.where(gaussian_k > 0.7, gaussian_k, torch.zeros_like(gaussian_k))
        gaussian_k = weight * gaussian_k
        mix_gaussian += gaussian_k
        i += 1

    mix_gaussian = mix_gaussian.unsqueeze(1).unsqueeze(2)  # [32, 1, 1, 512]
    mix_gaussian = mix_gaussian.expand(-1, 12, 30, -1)  
    combined_weight = 0.7 * gaussian_weight + mix_gaussian
    combined_weight = combined_weight / combined_weight.amax(dim=-1, keepdim=True)
    combined_weight = torch.where(combined_weight < 0.3, torch.zeros_like(combined_weight), combined_weight)

    return combined_weight

def custom_cross_attention_processing(
    scores,
    self_layer_idx,
    underscore_mask,
    t5_cand_pos_s,
    t5_cand_pos_e,
    initial_std,
    std_scaling_factor,
    topk,
    sim_word_idx,
    num_heads,
    seq_length,
    batch_size,
    key_length,
    device,
):
    """
    Apply custom cross-attention weighting with Gaussian and top-k based adjustment.
    This function computes combined attention weights for cross-attention, incorporating
    positional Gaussian weighting, underscore masking, and top-k important positions.
    """
    # 1. Generate Vanila Gaussian weights for each candidate span
    gaussian_weight = vanila_gaussian_weight(
        key_length, batch_size, t5_cand_pos_s, t5_cand_pos_e,
        self_layer_idx, initial_std, std_scaling_factor, device
    ).expand(batch_size, num_heads, seq_length, key_length)
    gaussian_weight = gaussian_weight.to(device)
    underscore_mask = underscore_mask.to(device)
    
    # 2. Compute mean attention map (average over decoder sequence positions)
    avg_attention_map = scores[:, :, 5:-2, :].mean(dim=2)
    masked_attention_map = avg_attention_map.clone()
    masked_attention_map.masked_fill_(underscore_mask.squeeze(1), float('-inf'))

    # 3. Identify top-k most attended document positions (tokens) for each head
    topk_values, topk_indices = torch.topk(masked_attention_map.view(batch_size, num_heads, -1), k=topk, dim=-1)
    sim_word_idx_tensor = torch.stack(sim_word_idx, dim=1).unsqueeze(1).expand(-1, num_heads, -1).to(device)
    combined_indices = torch.cat([topk_indices, sim_word_idx_tensor], dim=2)

    topk_mask = torch.zeros_like(avg_attention_map)
    topk_mask.view(batch_size, num_heads, -1).scatter_(-1, topk_indices, 1)
    extended_topk_mask = topk_mask.unsqueeze(2).expand(-1, -1, seq_length, -1)

    # 4. Generate gaussian mixture weights for the top-k candidates
    combined_weight = gaussian_mixture_weighting(
        scores, underscore_mask, 3, gaussian_weight, sim_word_idx_tensor
    )
    weighted_attention_map = torch.where(extended_topk_mask.bool(), torch.ones_like(gaussian_weight), combined_weight)

    # 5. Additional masking and thresholding
    weighted_attention_map = torch.where(underscore_mask, gaussian_weight, weighted_attention_map)

    # 6. Adjust the attention scores using the combined weights
    adjusted_scores = torch.where(
        scores < 0.,
        scores * (1 - weighted_attention_map),
        weighted_attention_map * scores
    )
    # 7. Compute final normalized attention weights
    attn_weights = nn.functional.softmax(adjusted_scores.float(), dim=-1).type_as(scores)
    return attn_weights


if is_cross_attention:
    # Extract required arguments from kwargs
    t5_cand_pos_s = kwargs.get("t5_cand_pos_s", None)
    t5_cand_pos_e = kwargs.get("t5_cand_pos_e", None)
    initial_std = kwargs.get("initial_std", None)
    std_scaling_factor = kwargs.get("std_scaling_factor", None)
    underscore_mask = kwargs.get("underscore_mask", None)
    topk = kwargs.get("topk", None)
    sim_word_idx = kwargs.get("sim_word_idx", None)
    batch_size, num_heads, seq_length, key_length = scores.shape

    if t5_cand_pos_s is None:
        raise ValueError("t5_cand_pos is required for cross-attention.")
    device = scores.device

    # Use the custom cross-attention weighting function
    attn_weights = custom_cross_attention_processing(
        scores, self.layer_idx, underscore_mask, t5_cand_pos_s, t5_cand_pos_e,
        initial_std, std_scaling_factor, topk, sim_word_idx, num_heads, seq_length,
        batch_size, key_length, device
    )

else:
    # Standard attention weights (softmax over scores)
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)

attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
