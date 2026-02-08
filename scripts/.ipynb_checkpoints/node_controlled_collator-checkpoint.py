from dataclasses import dataclass
from typing import Optional, Set
import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class NodeControlledMLMCollator(DataCollatorForLanguageModeling):
    """
    MLM collator that ONLY masks node-vocabulary tokens.

    - pretrained BERT subwords are NEVER masked
    - special tokens are NEVER masked
    - masking count is strictly controlled
    """

    node_token_ids: Set[int] = None

    exact_k_masks: Optional[int] = 1
    min_masks: int = 1
    max_masks: int = 2

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # ----- node-token mask -----
        if self.node_token_ids is None:
            raise ValueError("node_token_ids must be provided for NodeControlledMLMCollator")

        device = inputs.device
        node_token_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for tid in self.node_token_ids:
            node_token_mask |= (inputs == tid)

        # 마스킹 가능 위치:
        # 1) special token 아님
        # 2) node vocab 토큰임
        maskable = (~special_tokens_mask) & node_token_mask

        labels[:] = -100
        bsz, seqlen = inputs.shape
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)

        for i in range(bsz):
            candidates = torch.where(maskable[i])[0]
            if candidates.numel() == 0:
                continue

            if self.exact_k_masks is not None:
                k = min(int(self.exact_k_masks), candidates.numel())
            else:
                low = max(1, int(self.min_masks))
                high = max(low, int(self.max_masks))
                k = min(
                    torch.randint(low, high + 1, (1,), device=device).item(),
                    candidates.numel()
                )

            chosen = candidates[torch.randperm(candidates.numel(), device=device)[:k]]
            masked_indices[i, chosen] = True

        labels[masked_indices] = inputs[masked_indices]

        # 80% [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(inputs.shape, 0.8, device=device)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% random (node vocab 안에서만)
        indices_random = (
            torch.bernoulli(torch.full(inputs.shape, 0.5, device=device)).bool()
            & masked_indices
            & ~indices_replaced
        )

        node_ids = torch.tensor(list(self.node_token_ids), device=device)
        rand_idx = torch.randint(len(node_ids), inputs.shape, device=device)
        random_words = node_ids[rand_idx]
        inputs[indices_random] = random_words[indices_random]

        # 나머지 10% keep

        return inputs, labels


from dataclasses import dataclass
from typing import Set, Optional
import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class NodeControlledRandomMLMCollator(DataCollatorForLanguageModeling):

    node_token_ids: Set[int] = None  # 반드시 주입

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # 🔴 핵심 1: 기본 HF 확률 마스크
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # 🔴 핵심 2: node vocab이 아닌 곳은 마스킹 확률 0
        node_mask = torch.zeros_like(labels, dtype=torch.bool)
        for tid in self.node_token_ids:
            node_mask |= (inputs == tid)

        probability_matrix = probability_matrix * node_mask.float()
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
