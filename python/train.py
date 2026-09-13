"""
This script trains and tests a GraphSAGE model for node classification
on large graphs using GraphBolt dataloader.

Paper: [Inductive Representation Learning on Large Graphs]
(https://arxiv.org/abs/1706.02216)

Unlike previous dgl examples, we've utilized the newly defined dataloader
from GraphBolt. This example will help you grasp how to build an end-to-end
training pipeline using GraphBolt.

Before reading this example, please familiar yourself with graphsage node
classification by reading the example in the
`examples/core/graphsage/node_classification.py`. This introduction,
[A Blitz Introduction to Node Classification with DGL]
(https://docs.dgl.ai/tutorials/blitz/1_introduction.html), might be helpful.

If you want to train graphsage on a large graph in a distributed fashion,
please read the example in the `examples/distributed/graphsage/`.

This flowchart describes the main functional sequence of the provided example:
main
│
├───> OnDiskDataset pre-processing
│
├───> Instantiate SAGE model
│
├───> train
│     │
│     ├───> Get graphbolt dataloader (HIGHLIGHT)
│     │
│     └───> Training loop
│           │
│           ├───> SAGE.forward
│           │
│           └───> Validation set evaluation
│
└───> All nodes set inference & Test set evaluation
"""

import os, sys

os.environ['DGLBACKEND'] = 'pytorch'
# An experimental feature for CUDA allocations is turned on for better allocation
# pattern resulting in better memory usage for minibatch GNN training workloads.
# See https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf,
# and set the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
# if you want to disable it and set it True to acknowledge and disable the warning.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import argparse
import time

import dgl.graphbolt as gb
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from torch.cuda import Stream
from tqdm import tqdm
from pathlib import Path

import models
from data_loader import create_dataloader, MyDataLoader
from reporter import TimingReporter, DataLoaderTimingReporter


@torch.no_grad()
def evaluate(args, model, graph, features, itemset, num_classes):
    model.eval()
    y = []
    y_hats = []
    dataloader = create_dataloader(
        graph=graph,
        features=features,
        itemset=itemset,
        batch_size=args.batch_size,
        fanout=args.fanout,
        storage_device=args.storage_device,
        device=args.device,
        num_workers=args.num_workers,
        job="evaluate",
    )

    for data in tqdm(dataloader, "Evaluating"):
        x = data.node_features["feat"]
        y.append(data.labels)
        y_hats.append(model(data.blocks, x))

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(y),
        task="multiclass",
        num_classes=num_classes,
    )


def train(args, graph, features, train_set, valid_set, num_classes, model):
    for file in Path('.').glob('timestamp_*.txt'):
        os.remove(file)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    num_minibatches = (train_set._items[0].shape[0] + args.batch_size - 1) // args.batch_size
    dataload_reporter = DataLoaderTimingReporter(num_minibatches, args.num_workers)
    dataloader = create_dataloader(
        graph=graph,
        features=features,
        itemset=train_set,
        batch_size=args.batch_size,
        fanout=args.fanout,
        storage_device=args.storage_device,
        device=args.device,
        num_workers=args.num_workers,
        job="train",
        reporter=dataload_reporter,
        no_copy=True, # GPU transfer will be handled by MyDataLoader
        num_threads=args.num_threads_sample,
        num_minibatches=num_minibatches,
        prefetch_factor=None if(args.num_workers == 0) else args.prefetch_factor,
        pre_samples_root=args.pre_samples_root if(args.pre_sampling == "use_samples") else None,
        sample_mode=args.sample_mode,
        num_sampling_buffers=args.num_sampling_buffers,
    )

    if(args.storage_device == "cpu"):
        streams = [Stream(device=args.gpu_id) for i in range(3)]
        dataloader = MyDataLoader(dataloader, 2, args.device, streams)
    else:
        streams = [torch.cuda.default_stream(device=args.gpu_id)]

    compute_reporter = TimingReporter('compute', num_minibatches, args.num_workers)
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        num_features = 0
        for step, data in tqdm(enumerate(dataloader), "Training"):
        #for step, data in enumerate(dataloader):
            assert(data.minibatch_idx == step)
            compute_reporter.report_time('start', step)
            #if(prev_stream is not None):
            #    stream.wait_stream(prev_stream)
            current_stream = streams[step % len(streams)]
            with torch.cuda.stream(current_stream):
                x = data.node_features["feat"]
                y = data.labels
                y_hat = model(data.blocks, x) # blocks are computed at this point on the fly on the GPU
                loss = F.cross_entropy(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_features += x.shape[0]
            #current_stream.synchronize()
            #prev_stream = stream

            compute_reporter.report_time('end', step)

        t1 = time.time()
        # Evaluate the model.
        acc = evaluate(args, model, graph, features, valid_set, num_classes)
        assert(step + 1 == num_minibatches)
        print_stats(epoch, num_minibatches, total_loss, acc.item(), t1 - t0, num_features)

    del dataload_reporter
    torch.save(model.state_dict(), 'model_weights.pth')


def print_stats(epoch, num_minibatches, total_loss, accuracy, elapsed_time, num_features):
    with open("stats.txt", "wt") as file:
        for f in [sys.stdout, file]:
            print(
                f"Epoch {epoch:05d} | Loss {total_loss / num_minibatches:.4f} | "
                f"Accuracy {accuracy:.4f} | Time {elapsed_time:.4f}",
                file=f
            )
            print(
                f"{num_features} features transferred: "
                f"{num_features / num_minibatches:.2f} per minibatch, "
                f"{num_features / elapsed_time / 1000000:.2f} MIOPS",
                file=f
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script trains and tests a GraphSAGE model "
        "for node classification using GraphBolt dataloader."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimization.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for training."
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="10,10,10",
        help="Fan-out of neighbor sampling. It is IMPORTANT to keep len(fanout)"
        " identical with the number of layers in your model. Default: 10,10,10",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sage",
        choices=[
            "sage",
            "gcn",
            "gat",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=[
            "ogbn-arxiv",
            "ogbn-products",
            "ogbn-papers100M",
            "igb-hom-tiny",
            "igb-hom-small",
            "igb-hom-medium",
            "igb-hom-large",
            "igb-hom",
        ],
        help="The dataset we can use for node classification example. Currently"
        " ogbn-products, ogbn-arxiv, ogbn-papers100M and"
        " igb-hom-[tiny|small|medium|large] and igb-hom datasets are supported.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default='.',
    )
    parser.add_argument(
        "--mode",
        default="cpu-cuda", #"pinned-cuda",
        choices=["cpu-cpu", "cpu-cuda", "pinned-cuda", "cuda-cuda"],
        help="Dataset storage placement and Train device: 'cpu' for CPU and RAM,"
        " 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=-1,
        help="GPU ID (-1 for implicit GPU selection)",
    )
    parser.add_argument(
        "--sample-mode",
        default="sample_neighbor",
        choices=[
            "sample_neighbor",
            #"sample_layer_neighbor",
            "sample_neighbors_and_compact",
            "sample_neighbors_all",
        ],
        help="Sampling function",
    )
    parser.add_argument(
        "--num-sampling-buffers",
        type=int,
        default=0,
        help="The number of concurrently-sampled minibatches. 0 for serial (parallelized within each minibatch) execution",
    )
    parser.add_argument(
        "--disk-based-feature",
        default="none",
        choices=["none", "ssd", "spdk"],
    )
    parser.add_argument(
        "--fetch-mode",
        type=int,
        default=0,
        help="Feature fetch implementation"
    )
    parser.add_argument(
        "--num-contexts",
        type=int,
        default=0,
        help="Number of contexts to fetch disk-based features."
    )
    parser.add_argument(
        "--feature-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type of feature vectors. They will be cast to this type if they are not of this type."
    )
    parser.add_argument(
        "--model-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type of model. If this is different from feature dtype, feature dtype will be cast to the model dtype."
    )
    parser.add_argument(
        "--pre-sampling",
        type=str,
        default="none",
        choices=["none", "gen_samples", "use_samples"],
        help="whether to sample nodes in advance."
    )
    parser.add_argument(
        "--pre-samples-root",
        type=str,
        default='',
    )
    parser.add_argument(
        "--num-threads-sample",
        type=int,
        default=0,
        help="Number of threads to sample nodes."
    )
    parser.add_argument(
        "--num-threads-fetch",
        type=int,
        default=0,
        help="Number of threads to fetch features."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading.", # must be 0 for GPU sampling
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches loaded in advance by each worker.",
    )
    return parser.parse_args()


def save_sampled_minibatches(args, graph, train_set):
    args.pre_samples_root.mkdir(exist_ok=True)
    datapipe = gb.ItemSampler(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    datapipe = getattr(datapipe, args.sample_mode)(
        graph,
        args.fanout,
        overlap_fetch=False,
        asynchronous=False
    )
    dataloader = gb.DataLoader(datapipe)
    for i, data in enumerate(dataloader):
        torch.save(data, args.pre_samples_root.joinpath(f'minibatch{i:04d}.pt'))


def to_torch_dtype(dtype):
    if(dtype == "float32"):
        return torch.float32
    elif(dtype == "float16"):
        return torch.float16
    elif(dtype == "bfloat16"):
        if(torch.cuda.is_bf16_supported()):
            return torch.bfloat16
        else:
            print(f"bfloat16 not supported: original dtype {dtype} will be used")
            return None
    else:
        return None


def main(args):
    
    if(torch.cuda.is_available()):
        num_gpus = torch.cuda.device_count()
        print(f'Number of available CUDA devices: {num_gpus}')
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            print(f'Device {i}: {device_name}')
        if(args.gpu_id >= num_gpus):
            print(f'invalid GPU ID {args.gpu_id}, reverting to implicit selection')
            args.gpu_id = -1
    else:
        args.mode = "cpu-cpu"

    print(f"Training in {args.mode} mode.")
    args.storage_device, args.device = args.mode.split("-")
    if(args.gpu_id >= 0):
        args.device += f':{args.gpu_id}'
    args.device = torch.device(args.device)
    args.feature_dtype = to_torch_dtype(args.feature_dtype)
    args.model_dtype = to_torch_dtype(args.model_dtype)

    # Load and preprocess dataset.
    print("Loading data...")
    dataset = gb.BuiltinDataset(args.dataset, root=args.root)
    if(args.disk_based_feature != "none"):
        for feature in dataset.yaml_data["feature_data"]:
            feature_key = (feature["domain"], feature["type"], feature["name"])
            # Set the in_memory setting to False without modifying YAML file.
            if feature_key == ("node", None, "feat"):
                feature["in_memory"] = False
                feature["disk_type"] = args.disk_based_feature # add metadata
                # Replace the path by the one in half float
                if(args.feature_dtype == torch.float16):
                    path = Path(feature["path"])
                    path = path.with_name(path.stem + '-half.npy')
                    full_path = Path(dataset._dataset_dir, path)
                    if(full_path.exists()):
                        feature["path"] = str(path)
                    else:
                        print(f'{full_path} does not exist')
                        exit(1)

    if(args.num_threads_sample <= 0):
        args.num_threads_sample = None
    if(args.num_threads_fetch <= 0):
        args.num_threads_fetch = None
    if(args.num_contexts <= 0):
        args.num_contexts = None
    dataset = dataset.load(num_threads=args.num_threads_fetch,
                           num_contexts=args.num_contexts,
                           index_select_mode=args.fetch_mode)
    if(not args.pre_samples_root):
        args.pre_samples_root = Path(dataset._dataset_dir, 'sampled')

    # convert the data type of in-memory feature
    if(args.disk_based_feature == "none"):
        torch_based_feature = dataset.feature[("node", None, "feat")]
        if(args.feature_dtype != torch_based_feature._tensor.dtype):
            feature_tensor = torch_based_feature.read()
            torch_based_feature.update(feature_tensor.to(dtype=args.feature_dtype))

    # Move the dataset to the selected storage.
    if args.storage_device == "pinned":
        with torch.cuda.device(args.device):
            graph = dataset.graph.pin_memory_()
            features = dataset.feature.pin_memory_()
    else:
        graph = dataset.graph.to(args.storage_device)
        features = dataset.feature
        if(args.disk_based_feature == "none"):
            features = features.to(args.storage_device)
    # graph = dgl.add_self_loop(graph) # AttributeError: 'FusedCSCSamplingGraph' object has no attribute 'to_canonical_etype'

    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    all_nodes_set = dataset.all_nodes_set
    args.fanout = list(map(int, args.fanout.split(",")))

    num_classes = dataset.tasks[0].metadata["num_classes"]

    in_size = features.size("node", None, "feat")[0]
    hidden_size = 256
    out_size = num_classes

    print(f'train set size: {train_set._items[0].shape[0]}')
    print(f'valid set size: {valid_set._items[0].shape[0]}')
    print(f'test  set size: {test_set._items[0].shape[0]}')

    if(args.pre_sampling == "gen_samples"):
        save_sampled_minibatches(args, graph, train_set)
        exit(0)

    if(args.model == "sage"):
        model = models.SAGE(in_size, hidden_size, out_size, len(args.fanout), args.feature_dtype)
    elif(args.model == "gcn"):
        model = models.GCN(in_size, hidden_size, out_size, len(args.fanout), args.feature_dtype)
    elif(args.model == "gat"):
        hidden_size = 64
        model = models.GAT(in_size, hidden_size, out_size, len(args.fanout), args.feature_dtype)
    assert len(args.fanout) == len(model.layers)
    model = model.to(args.device)
    if(args.model_dtype is not None):
        model = model.to(dtype=args.model_dtype)


    # Model training.
    print("Training...")
    train(args, graph, features, train_set, valid_set, num_classes, model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
