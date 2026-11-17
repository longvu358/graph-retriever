import torch

from graph_retriever.model import HGRetriever


def _model(encoder: str) -> HGRetriever:
    torch.manual_seed(0)
    P = torch.eye(6) / 2  # any fixed operator works for this check
    model = HGRetriever(6, {"area": P}, dim=8, encoder=encoder, residual=True).eval()
    return model


def test_gru_encoder_ignores_right_padding():
    model = _model("gru")
    short = torch.tensor([[1, 2]])
    short_mask = torch.tensor([[True, True]])
    padded = torch.tensor([[1, 2, 0, 0, 0]])
    padded_mask = torch.tensor([[True, True, False, False, False]])
    with torch.no_grad():
        a = model(short, short_mask)
        b = model(padded, padded_mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_attn_encoder_ignores_right_padding():
    model = _model("attn")
    short = torch.tensor([[1, 2]])
    short_mask = torch.tensor([[True, True]])
    padded = torch.tensor([[1, 2, 0, 0]])
    padded_mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        a = model(short, short_mask)
        b = model(padded, padded_mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_no_graph_control_runs_without_channels():
    torch.manual_seed(0)
    model = HGRetriever(6, {}, dim=8, encoder="gru", residual=True).eval()
    prefix = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[True, True, True]])
    with torch.no_grad():
        scores = model(prefix, mask)
    assert scores.shape == (1, 6)


def test_session_edge_ignores_right_padding():
    torch.manual_seed(0)
    P = torch.eye(6) / 2
    model = HGRetriever(6, {"area": P}, dim=8, encoder="gru", residual=True,
                        session_edge=True).eval()
    short = (torch.tensor([[1, 2]]), torch.tensor([[True, True]]))
    padded = (torch.tensor([[1, 2, 0, 0]]), torch.tensor([[True, True, False, False]]))
    with torch.no_grad():
        assert torch.allclose(model(*short), model(*padded), atol=1e-6)


def test_time_ctx_changes_scores_and_is_optional():
    torch.manual_seed(0)
    P = torch.eye(6) / 2
    model = HGRetriever(6, {"area": P}, dim=8, encoder="gru", residual=True, n_time=12).eval()
    prefix = torch.tensor([[1, 2]])
    mask = torch.tensor([[True, True]])
    with torch.no_grad():
        a = model(prefix, mask, torch.tensor([0]))
        b = model(prefix, mask, torch.tensor([5]))
        none = model(prefix, mask)
    assert not torch.allclose(a, b)
    assert none.shape == a.shape


def test_ssl_loss_is_finite_and_uses_prefix_only():
    torch.manual_seed(0)
    P = torch.eye(6) / 2
    clique = torch.full((6, 6), 1 / 6)
    model = HGRetriever(6, {"area": P}, dim=8, encoder="gru", residual=True,
                        clique_op=clique).eval()
    prefix = torch.tensor([[1, 2], [3, 4]])
    mask = torch.tensor([[True, True], [True, True]])
    with torch.no_grad():
        loss = model.ssl_loss(prefix, mask)
        padded = model.ssl_loss(torch.tensor([[1, 2, 0], [3, 4, 0]]),
                                torch.tensor([[True, True, False], [True, True, False]]))
    assert torch.isfinite(loss)
    assert torch.allclose(loss, padded, atol=1e-6)
