import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator

from models import T1
from hyper import BATCH, DEVICE, TOLERANCE, make_optimiser

if __name__ == '__main__':
    torch.manual_seed(0)

    dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="datasets")
    dataset.load_val_ns()
    dataset.load_test_ns()
    evaluator = Evaluator(name="tgbl-wiki")
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

    model = T1(total_nodes, total_events).to(DEVICE)
    optimiser = make_optimiser(model)
    writer = SummaryWriter()
    total_examples = 0
    epoch = 0

    best_validation = float('-inf')
    tolerance = TOLERANCE

    def test(event, validation: bool = False) -> float :
        root = src[event]
        pos_dst = dst[event]
        all_dst = torch.tensor(
            [dst[event]] +
            dataset.negative_sampler.query_batch(
                src[event:event + 1],
                dst[event:event + 1],
                ts[event:event + 1],
                split_mode='val' if validation else 'test'
            )[0]
        )
        all_src = root.repeat(all_dst.shape)
        with torch.no_grad():
            hs = model.embed(src, dst, ts, event)
            y = model.predict_link(hs[-1], all_src, all_dst)
        model.remember(hs, root, pos_dst, event)
        return evaluator.eval({
            'y_pred_pos': y[0],
            'y_pred_neg': y[1:].squeeze(),
            'eval_metric': [dataset.eval_metric]
        })[dataset.eval_metric]


    while True:
        print(f"epoch: {epoch}")
        # train
        model.train()
        for event in range(num_train):
            hs = model.embed(src, dst, ts, event)
            root = src[event]
            pos_dst = dst[event]
            neg_dst = torch.randint(min_dst, max_dst + 1, ())
            loss = F.binary_cross_entropy_with_logits(
                model.predict_link(
                    hs[-1],
                    torch.tensor([root, root]),
                    torch.tensor([neg_dst, pos_dst])
                ),
                torch.tensor([[0.], [1.]], device=DEVICE)
            )
            loss.backward()
            writer.add_scalar('loss', loss.detach(), total_examples)
            total_examples += 1
            if total_examples % BATCH == 0:
                optimiser.step()
                optimiser.zero_grad()

            model.remember(hs, root, pos_dst, event)

        # validate
        model.eval()
        validation_metric = 0
        for event in range(num_train, num_train + num_val):
            validation_metric += test(event, validation=True)

        validation_metric /= num_val
        print(f"validation: {validation_metric:.5f}")
        writer.add_scalar('validation', validation_metric, epoch)

        if validation_metric >= best_validation:
            best_validation = validation_metric
            tolerance = TOLERANCE
            print("best so far, saving to checkpoint.pt")
            torch.save(model, 'checkpoint.pt')
        else:
            print(f"not better, tolerance = {tolerance}")
            tolerance -= 1

        if tolerance < 0:
            print("failed to improve, exit training")
            break

        epoch += 1

    model = torch.load('checkpoint.pt')
    model.eval()
    # "rehydrate" model with events
    for event in range(num_train + num_val):
        with torch.no_grad():
            hs = model.embed(src, dst, ts, event)
            model.remember(hs, src[event], dst[event], event)

    test_metric = 0
    for event in range(num_train + num_val, num_train + num_val + num_test):
        test_metric += test(event)
    test_metric /= num_test
    print(f"test: {test_metric:.5f}")
