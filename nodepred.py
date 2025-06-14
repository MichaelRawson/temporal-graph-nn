import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator

from models import Model
from hyper import DEVICE, PATIENCE, make_optimiser

def batches(dataset):
    """iterator of [begin, end) batches"""

    i = 0
    for time, y_map in dataset.dataset.label_dict.items():
        j = i + 1
        offset = 1
        while dataset.ts[j] < time:
            # don't know where we next meet a boundary
            # jump ahead by exponentially increasing amount
            if j + offset < len(dataset.ts) and dataset.ts[j + offset] < time:
                j += offset
                offset *= 2
            else:
                # or back off a bit
                offset //= 2
                j += 1

        projection = list(y_map.keys())
        y = torch.tensor(np.array([y_map[k] for k in projection]), dtype=torch.float)
        projection = torch.tensor(projection, dtype=torch.long)
        yield i, j, projection, y
        i = j

if __name__ == '__main__':
    import sys
    _, flavour, data_name = sys.argv

    torch.manual_seed(0)

    dataset = PyGNodePropPredDataset(name=data_name, root="datasets")
    dataset.process_data()
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

    model = Model(flavour, total_nodes, total_events, dataset.num_classes).to(DEVICE)
    optimiser = make_optimiser(model)
    writer = SummaryWriter()
    total_examples = 0
    epoch = 0

    best_validation = float('+inf')
    patience = PATIENCE

    def forward(current, after, project, y) -> Tensor:
        """do the forward pass and have `model` remember embeddings"""

        hs = model.embed(src, dst, ts, after)
        root = src[current:after]
        pos_dst = dst[current:after]
        embedding = hs[-1][project]
        prediction = model.predict_node(embedding)
        loss = F.cross_entropy(prediction, y.to(DEVICE))
        model.remember(hs, root, pos_dst, current)
        return loss


    while True:
        print(f"epoch: {epoch}")

        # train
        batch_iterator = batches(dataset)
        model.train()
        for current, after, projection, y in batch_iterator:
            loss = forward(current, after, projection, y)
            loss.backward()
            writer.add_scalar('loss', loss.detach(), total_examples)
            optimiser.step()
            optimiser.zero_grad()
            total_examples += after - current
            if after > num_train:
                break

        # validate
        model.eval()
        validation_loss = 0
        validation_steps = 0
        for current, after, projection, y in batch_iterator:
            with torch.no_grad():
                validation_loss += forward(current, after, projection, y).detach()
                validation_steps += 1
            if after > num_train + num_val:
                break

        validation_loss /= validation_steps
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
    batch_iterator = batches(dataset)
    # "rehydrate" model with events
    for current, after, _, _ in batch_iterator:
        with torch.no_grad():
            hs = model.embed(src, dst, ts, current)
            model.remember(hs, src[current:after], dst[current:after], current)
        if after > num_train + num_val:
            break

    test_metric = 0
    for current, after, projection, y in batch_iterator:
        with torch.no_grad():
            hs = model.embed(src, dst, ts, current)
            embedding = hs[-1][projection]
            prediction = model.predict_node(embedding)

        prediction = prediction.cpu().numpy()
        y = y.cpu().numpy()

        input_dict = {
            "y_true": y,
            "y_pred": prediction,
            "eval_metric": [dataset.eval_metric],
        }
        test_metric += evaluator.eval(input_dict)[dataset.eval_metric]
        model.remember(hs, src[current:after], dst[current:after], current)

    test_metric /= num_test
    print(f"test: {test_metric:.5f}")
