import torch.nn.functional as F


def kl_distill_loss(student_logits, teacher_logits, temperature=2.0):
    """
    KL(student || teacher target distribution).

    student_logits: [B, T, V]
    teacher_logits: [B, T, V]
    """
    temp = temperature

    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)

    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temp * temp)

    return loss
