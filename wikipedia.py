import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from models import T1
from hyper import BATCH

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    torch.manual_seed(0)

    dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="datasets")
    total_nodes = max(int(dataset.src.max()), int(dataset.dst.max().item())) + 1
    total_events = len(dataset.ts)
    min_dst = int(dataset.dst.min())
    max_dst = int(dataset.dst.max())

    src = dataset.src.to(DEVICE)
    dst = dataset.dst.to(DEVICE)
    ts = dataset.ts.to(DEVICE)

    model = T1(total_nodes, total_events).to(DEVICE)
    optimiser = Adam(model.parameters())
    writer = SummaryWriter()
    total_examples = 0
    while True:
        for event in range(total_events - 1):
            h = model.embed(src, dst, ts, event)
            root = src[event + 1]
            pos_dst = dst[event + 1]
            neg_dst = torch.randint(min_dst, max_dst + 1, ())
            loss = F.binary_cross_entropy_with_logits(
                model.predict_link(h, root, pos_dst),
                torch.tensor([1.], device=DEVICE)
            ) + F.binary_cross_entropy_with_logits(
                model.predict_link(h, root, neg_dst),
                torch.tensor([0.], device=DEVICE)
            )
            loss.backward()
            writer.add_scalar('loss', loss.detach(), total_examples)
            total_examples += 1
            if total_examples % BATCH == 0:
                optimiser.step()
                optimiser.zero_grad()
