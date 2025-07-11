from src.layers.Dismult import DistMultDecoder
from src.layers.GATDecoder import GATDecoder
from src.layers.GATEncoder import GATEncoder
from src.layers.GCNDecoder import GCNDecoder
from src.layers.GCNEncoder import GCNEncoder
from src.layers.MLPDecoder import MLPDecoder
from src.layers.RGCNDecoder import RGCNDecoder
from src.layers.RGCNEncoder import RGCNEncoder
from src.layers.TransGCNDecoder import TransGCNDecoder
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


def instantiate_decoder(config_, data, encoder):
    """
    Instancie un décodeur basé sur la configuration
    """
    decoder_type = config_.get("classifier_decoder", config_.get("decoder_type"))
    device = torch.device(config_["device"])
    alpha = config_.get("alpha", 0.01)
    num_bases = config_.get("num_bases", None)
    use_edges_info = config_.get("use_edges_info", False)
    msg_sens = config_.get("message_sens", "source_to_target")

    if decoder_type == "GCN":
        decoder = GCNDecoder(encoder, data, alpha, message_sens=msg_sens).to(device)

    elif decoder_type == "RGCN":
        decoder = RGCNDecoder(encoder, data, num_bases, alpha, message_sens=msg_sens).to(device)

    elif decoder_type == "MLP":
        decoder = MLPDecoder(encoder, data, alpha).to(device)

    elif decoder_type in ["TransGCN_conv", "TransGCN_attn"]:
        variant = "conv" if "conv" in decoder_type else "attn"
        decoder = TransGCNDecoder(
            encoder, data, alpha, dropout=0.3,
            kg_score_fn='TransE', variant=variant,
            use_edges_info=use_edges_info
        ).to(device)

    elif decoder_type in ["RotatEGCN_conv", "RotatEGCN_attn"]:
        variant = "conv" if "conv" in decoder_type else "attn"
        decoder = TransGCNDecoder(
            encoder, data, alpha, dropout=0.3,
            kg_score_fn='RotatE', variant=variant,
            use_edges_info=use_edges_info
        ).to(device)

    elif decoder_type == "GAT":
        decoder = GATDecoder(encoder, data, heads=4, alpha=alpha, dropout=0.3).to(device)

    elif decoder_type == "Dismult":
        out_channels = config_.get("encoder_out_channels", config_.get("out_channels", [64]))
        if isinstance(out_channels, list):
            out_channels = out_channels[-1]
        decoder = DistMultDecoder(data.num_edge_types, out_channels).to(device)

    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    return decoder