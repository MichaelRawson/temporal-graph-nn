from typing import Generator

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator

from models import Model
from hyper import BATCH, DEVICE, PATIENCE, make_optimiser

def batches(start: int, end: int) -> Generator[tuple[int, int], None, None]:
    """iterator of [begin, end) batches"""

    old = None
    current = start
    while current < end:
        old = current
        current += BATCH
        if current > end:
            current = end
        yield old, current

if __name__ == '__main__':
    import sys
    _, flavour, data_name = sys.argv

    torch.manual_seed(0)

    dataset = PyGLinkPropPredDataset(name=data_name, root="datasets")
    dataset.load_val_ns()
    dataset.load_test_ns()
    evaluator = Evaluator(name=data_name)
    total_nodes = max(int(dataset.src.max()), int(dataset.dst.max().item())) + 1
    total_events = len(dataset.ts)
    num_train = int(dataset.train_mask.sum())
    num_val = int(dataset.val_mask.sum())
    num_test = int(dataset.test_mask.sum())
    min_dst = int(dataset.dst.min())
    max_dst = int(dataset.dst.max())

    src = dataset.src.to(DEVICE)
    dst = dataset.dst.to(DEVICE)
    ts = dataset.ts.to(DEVICE)

    model = Model(flavour, total_nodes, total_events).to(DEVICE)
    optimiser = make_optimiser(model)
    writer = SummaryWriter()
    total_examples = 0
    epoch = 0

    best_validation = float('+inf')
    patience = PATIENCE

    zeros = torch.zeros(BATCH, device=DEVICE)
    ones = torch.ones(BATCH, device=DEVICE)
    def forward(current: int, after: int) -> Tensor:
        """do the forward pass and have `model` remember embeddings"""

        batch = after - current
        hs = model.embed(src, dst, ts, current)
        root = src[current:after]
        pos_dst = dst[current:after]
        neg_dst = torch.randint(min_dst, max_dst + 1, (batch,), device=DEVICE)
        loss = F.binary_cross_entropy_with_logits(
            model.predict_link(
                hs[-1],
                torch.cat((root, root)),
                torch.cat((neg_dst, pos_dst))
            ),
            torch.cat((zeros[:batch], ones[:batch]))
        )
        model.remember(hs, root, pos_dst, current)
        return loss


    while True:
        print(f"epoch: {epoch}")

        # train
        model.train()
        for current, after in batches(0, num_train):
            loss = forward(current, after)
            loss.backward()
            writer.add_scalar('loss', loss.detach(), total_examples)
            optimiser.step()
            optimiser.zero_grad()
            total_examples += after - current

        # validate
        model.eval()
        validation_loss = 0
        for current, after in batches(num_train, num_train + num_val):
            with torch.no_grad():
                validation_loss += forward(current, after).detach()

        validation_loss /= (1 + num_val / BATCH)
        print(f"validation: {validation_loss:.5f}")
        writer.add_scalar('validation', validation_loss, epoch)

        if validation_loss < best_validation:
            best_validation = validation_loss
            patience = PATIENCE
            print("best so far, saving to checkpoint.pt")
            torch.save(model, 'checkpoint.pt')
        else:
            print(f"not better, patience = {patience}")
            patience -= 1

        if patience < 0:
            print("failed to improve, exit training")
            break

        epoch += 1

    model = torch.load('checkpoint.pt')
    model.eval()
    # "rehydrate" model with events
    for current, after in batches(0, num_train + num_val):
        with torch.no_grad():
            hs = model.embed(src, dst, ts, current)
            model.remember(hs, src[current:after], dst[current:after], current)

    test_metric = 0
    for current, after in batches(num_train + num_val, num_train + num_val + num_test):
        root = src[current:after]
        pos_dst = dst[current:after]
        with torch.no_grad():
            hs = model.embed(src, dst, ts, current)

        for i, neg_batch in enumerate(dataset.negative_sampler.query_batch(
            root,
            pos_dst,
            ts[current:after],
            split_mode='test'
        )):
            all_dst = torch.tensor(
                [pos_dst[i]] +
                neg_batch
            )
            all_src = root[i].repeat(all_dst.shape)
            with torch.no_grad():
                y = model.predict_link(hs[-1], all_src, all_dst)
            test_metric += evaluator.eval({
                'y_pred_pos': y[0],
                'y_pred_neg': y[1:].squeeze(),
                'eval_metric': [dataset.eval_metric]
            })[dataset.eval_metric]
        model.remember(hs, src[current:after], dst[current:after], current)

    test_metric /= num_test
    print(f"test: {test_metric:.5f}")
