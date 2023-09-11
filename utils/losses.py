import torch
from torch.nn.functional import cross_entropy


'''
Refer to below for more.
https://github.com/AberHu/Knowledge-Distillation-Zoo
'''

def knowledge_distillation_loss(outs_student: torch.Tensor, outs_teacher: torch.Tensor, temperature: float, alpha: float, **kwargs):
    ''' Geoffrey Hinton, Oriol Vinyals, Jeff Dean "Distilling the Knowledge in a Neural Network" NIPS 2014. '''
    outs_student = torch.softmax(outs_student / temperature, dim=1)
    outs_teacher = torch.softmax(outs_teacher / temperature, dim=1)
    
    ce_loss = cross_entropy(outs_student, outs_teacher)
    kld_loss = torch.kl_div(outs_student.log(), outs_teacher, reduction='batchmean')
    
    return alpha * ce_loss + (1 - alpha) * (temperature**2) * kld_loss


def multi_distillation_loss(outs_student: torch.Tensor, outs_teacher: torch.Tensor, 
                            temperatures: list[float], num_classes: int, **kwargs):
    ''' Ying Jin, Jiaqi Wang, Dahua Li "Multi-Level Logit Distillation" CVPR 2023. '''
    batch_size = outs_student.size(0)
    loss_total = 0
    for t in temperatures:
        p_stu = torch.softmax(outs_student / t)  # ............| B x C
        p_tea = torch.softmax(outs_teacher / t)  # ............| B x C
        l_ins = torch.kl_div(p_tea, p_stu)
        G_stu = torch.mm(p_stu, p_stu.t())  # .................| B x B
        G_tea = torch.mm(p_tea, p_tea.t())  # .................| B x B
        l_batch = ((G_stu - G_tea) ** 2).sum() / batch_size
        M_stu = torch.mm(p_stu.t(), p_stu)  # .................| C x C
        M_tea = torch.mm(p_tea.t(), p_tea)  # .................| C x C
        l_class = ((M_stu - M_tea) ** 2).sum() / num_classes
        loss_total += l_ins + l_batch + l_class
    return loss_total
