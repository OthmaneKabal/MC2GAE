from src.layers.GATEncoder import GATEncoder
from src.layers.GCNEncoder import GCNEncoder
from src.layers.RGCNEncoder import RGCNEncoder
from src.layers.TransGCNEncoder import TransGCNEncoder
import torch

def instantiate_encoder(config_, data):
    encoder_type = config_["classifier_encoder"]
    out_channels = config_["encoder_out_channels"]
    device = torch.device(config_["device"])
    num_layers = config_["num_layers"]
    use_edges_info = config_.get("use_edges_info", False)
    num_bases = config_.get("num_bases", None)
    msg_sens = config_.get("message_sens", "source_to_target")

    if encoder_type == "GCN":
        encoder = GCNEncoder(data, out_channels, num_layers,
                             message_sens=msg_sens).to(device)

    elif encoder_type == "RGCN":
        encoder = RGCNEncoder(data, out_channels, num_layers, num_bases,
                              message_sens=msg_sens).to(device)

    elif encoder_type in ["TransGCN_conv", "TransGCN_attn"]:
        variant = "conv" if "conv" in encoder_type else "attn"
        encoder = TransGCNEncoder(
            data, out_channels, num_layers, dropout=0.2,
            kg_score_fn='TransE', variant=variant,
            use_edges_info=use_edges_info, activation='relu',
            bias=False
        ).to(device)

    elif encoder_type in ["RotatEGCN_conv", "RotatEGCN_attn"]:
        variant = "conv" if "conv" in encoder_type else "attn"
        encoder = TransGCNEncoder(
            data, out_channels, num_layers, dropout=0.2,
            kg_score_fn='RotatE', variant=variant,
            use_edges_info=use_edges_info, activation='relu',
            bias=False
        ).to(device)

    elif encoder_type == "GAT":
        encoder = GATEncoder(data, out_channels, num_layers).to(device)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder
