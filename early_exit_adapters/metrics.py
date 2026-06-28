import math

import torch
import torch.nn.functional as F


def compute_training_metrics(student_logits, teacher_logits, labels, top_k=5):
    """
    student_logits: [B, T-1, V]
    teacher_logits: [B, T-1, V]
    labels:         [B, T-1]
    """
    with torch.no_grad():
        batch_size, seq_minus_one, vocab_size = student_logits.shape

        student_logits_f = student_logits.float()
        teacher_logits_f = teacher_logits.float()

        student_log_probs = F.log_softmax(student_logits_f, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits_f, dim=-1)

        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        kl_to_teacher = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1).mean()

        ce = F.cross_entropy(
            student_logits_f.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="mean",
        )

        gt_probs = student_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        teacher_top1 = teacher_logits_f.argmax(dim=-1)
        student_top1 = student_logits_f.argmax(dim=-1)

        top1_agreement = (teacher_top1 == student_top1).float().mean()

        teacher_topk = teacher_logits_f.topk(top_k, dim=-1).indices
        student_topk = student_logits_f.topk(top_k, dim=-1).indices

        topk_overlap = (
            teacher_topk.unsqueeze(-1) == student_topk.unsqueeze(-2)
        ).any(dim=-1).float().mean()

        accept_proxy_exact = torch.minimum(
            teacher_probs,
            student_probs,
        ).sum(dim=-1).mean()

        sampled = torch.multinomial(
            student_probs.reshape(-1, vocab_size),
            num_samples=1,
        ).reshape(batch_size, seq_minus_one)

        p_token = teacher_probs.gather(
            dim=-1,
            index=sampled.unsqueeze(-1),
        ).squeeze(-1)

        q_token = student_probs.gather(
            dim=-1,
            index=sampled.unsqueeze(-1),
        ).squeeze(-1)

        accept_proxy_sampled = torch.minimum(
            torch.ones_like(p_token),
            p_token / torch.clamp(q_token, min=1e-12),
        ).mean()

        return {
            "kl_to_teacher": float(kl_to_teacher.item()),
            "ce": float(ce.item()),
            "perplexity": float(math.exp(min(ce.item(), 20))),
            "mean_gt_prob": float(gt_probs.mean().item()),
            "top1_teacher_agreement": float(top1_agreement.item()),
            f"top{top_k}_overlap": float(topk_overlap.item()),
            "accept_proxy_exact": float(accept_proxy_exact.item()),
            "accept_proxy_sampled": float(accept_proxy_sampled.item()),
        }
