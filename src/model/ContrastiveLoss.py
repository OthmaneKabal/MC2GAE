import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.5):
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        # Normalisation des embeddings
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        # Calcul de la similarité cosinus
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        # Fonction pour appliquer l'exponentielle avec la température
        f = lambda x: torch.exp(x / self.tau)

        # Similarité cosinus entre les embeddings dans la même vue (référence)
        refl_sim = f(self.sim(z1, z1))

        # Similarité cosinus entre les embeddings des vues différentes
        between_sim = f(self.sim(z1, z2))

        # Calcul de la perte InfoNCE
        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        ).mean()





    def contrastive_loss(self, H_1: torch.Tensor, H_2: torch.Tensor):
        # Calcul de la perte semi-contrastive entre H_1 et H_2
        l1 = self.semi_loss(H_1, H_2)
        l2 = self.semi_loss(H_2, H_1)
        # Moyenne des deux pertes pour la symétrie
        return (l1 + l2) / 2


