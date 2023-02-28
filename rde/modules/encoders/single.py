import torch
import torch.nn as nn

from rde.modules.common.layers import AngularEncoding


class PerResidueEncoder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        infeat_dim = feat_dim + self.dihed_embed.get_out_dim(6) # Phi, Psi, Chi1-4
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, phi, phi_mask, psi, psi_mask, chi, chi_mask, mask_residue):
        """
        Args:
            aa: (N, L)
            phi, phi_mask: (N, L)
            psi, psi_mask: (N, L)
            chi, chi_mask: (N, L, 4)
            mask_residue: (N, L)
        """
        N, L = aa.size()

        # Amino acid identity features
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Dihedral features
        dihedral = torch.cat(
            [phi[..., None], psi[..., None], chi], 
            dim=-1
        ) # (N, L, 6)
        dihedral_mask = torch.cat([
            phi_mask[..., None], psi_mask[..., None], chi_mask], 
            dim=-1
        ) # (N, L, 6)
        dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 6, feat)
        dihedral_feat = dihedral_feat.reshape(N, L, -1)

        # Mix
        out_feat = self.mlp(torch.cat([aa_feat, dihedral_feat], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat
