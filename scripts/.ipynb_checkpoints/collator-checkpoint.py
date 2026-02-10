from dataclasses import dataclass
from typing import Set, Optional
import torch
from transformers import DataCollatorForLanguageModeling

class NodeOnlyMLMCollator(DataCollatorForLanguageModeling):
    """
    - node_token_ids 위치만 마스킹 후보
    - mlm_probability로 "몇 개를" 뽑을지 결정
    - replacement 비중 조절 가능 (기본: 100% [MASK])
    """
    def __init__(
        self,
        tokenizer,
        node_token_ids: Set[int],
        mlm_probability: float = 0.15,
        mask_replace_prob: float = 1.0,
        random_replace_prob: float = 0.0,
    ):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        if not node_token_ids:
            raise ValueError("node_token_ids is empty")
        if mask_replace_prob + random_replace_prob > 1.0 + 1e-9:
            raise ValueError("mask_replace_prob + random_replace_prob must be <= 1.0")

        self.node_token_ids = set(int(x) for x in node_token_ids)
        self.mask_replace_prob = float(mask_replace_prob)
        self.random_replace_prob = float(random_replace_prob)

        # for random replacement
        self._node_ids_sorted = None

    def _get_node_ids_tensor(self, device):
        if self._node_ids_sorted is None:
            self._node_ids_sorted = torch.tensor(sorted(self.node_token_ids), dtype=torch.long)
        return self._node_ids_sorted.to(device)

    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None):
        labels = inputs.clone()
        device = inputs.device

        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        node_ids = self._get_node_ids_tensor(device)

        # ✅ fast node-only candidate mask
        # torch.isin exists in recent torch; if not, replace with a lookup-table approach.
        node_mask = torch.isin(inputs, node_ids)

        prob = torch.full(labels.shape, float(self.mlm_probability), device=device)
        prob = prob * node_mask.float()
        prob.masked_fill_(special_tokens_mask, 0.0)

        masked_indices = torch.bernoulli(prob).bool()
        labels[~masked_indices] = -100

        # 1) [MASK]
        replace_mask = (torch.rand(labels.shape, device=device) < self.mask_replace_prob) & masked_indices
        inputs[replace_mask] = self.tokenizer.mask_token_id

        # 2) random replacement (optional)
        if self.random_replace_prob > 0:
            remaining = masked_indices & ~replace_mask
            replace_rand = (torch.rand(labels.shape, device=device) < self.random_replace_prob) & remaining
            if replace_rand.any():
                rand_idx = torch.randint(len(node_ids), labels.shape, device=device)
                random_words = node_ids[rand_idx]
                inputs[replace_rand] = random_words[replace_rand]

        # keep: do nothing
        return inputs, labels



class NodeControlledMLMCollator(DataCollatorForLanguageModeling):
    """
    각 샘플당 정확히 1개의 node token 위치를 선택해서 MLM 타겟으로 만든다.
    기본: 100% [MASK], random replacement 없음.
    """
    def __init__(
        self,
        tokenizer,
        node_token_ids: Set[int],
        mlm: bool = True,
        mlm_probability: float = 0.15,  # (여기선 의미 거의 없음. parent 요구로 둠)
        exact_k_masks: int = 1,
        mask_replace_prob: float = 1.0,
        random_replace_prob: float = 0.0,
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

        if not node_token_ids:
            raise ValueError("node_token_ids required")
        if mask_replace_prob + random_replace_prob > 1.0 + 1e-9:
            raise ValueError("mask_replace_prob + random_replace_prob must be <= 1.0")

        self.node_token_ids = set(int(x) for x in node_token_ids)
        self.exact_k_masks = int(exact_k_masks)
        self.mask_replace_prob = float(mask_replace_prob)
        self.random_replace_prob = float(random_replace_prob)

        # cached sorted node ids for sampling
        self._node_ids_sorted_cpu = torch.tensor(sorted(self.node_token_ids), dtype=torch.long)

    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None):
        device = inputs.device
        node_ids = self._node_ids_sorted_cpu.to(device)

        # labels: default ignore everywhere
        labels = torch.full_like(inputs, -100)

        # node positions
        node_pos_mask = torch.isin(inputs, node_ids)

        B, L = inputs.shape
        K = self.exact_k_masks
        
        for b in range(B):
            pos = torch.nonzero(node_pos_mask[b], as_tuple=False).flatten()
            if pos.numel() == 0:
                continue

            k = min(K, pos.numel())
            chosen = pos[torch.randperm(pos.numel(), device=device)[:k]]

            for j in chosen.tolist():
                labels[b, j] = inputs[b, j]
    
                r = torch.rand((), device=device).item()
                if r < self.mask_replace_prob:
                    inputs[b, j] = self.tokenizer.mask_token_id
                elif r < self.mask_replace_prob + self.random_replace_prob:
                    inputs[b, j] = node_ids[
                        torch.randint(len(node_ids), (1,), device=device).item()
                    ]
                else:
                    pass  # keep original

        return inputs, labels
