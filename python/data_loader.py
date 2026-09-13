import time
from collections import deque
from typing import List, Deque, Iterator
from pathlib import Path
import threading

import torch
from torch.cuda import Stream

import dgl.graphbolt as gb

from reporter import TimingReporter


PRODUCER_SLEEP_INTERVAL = 0.0001
CONSUMER_SLEEP_INTERVAL = 0.0001


class _MyPrefetchData:
    def __init__(self, dataloader, buffer_size: int, device: torch.device, streams: List[Stream]):
        self.run_prefetcher: bool = True
        self.prefetch_buffer: Deque = deque()
        self.buffer_size: int = buffer_size
        self.dataloader = dataloader
        self.stop_iteration: bool = False
        self.paused: bool = False
        self.device: torch.device = device
        self.streams: List[Stream] = streams



class MyDataLoader:

    def __init__(self, dataloader, buffer_size: int, device: torch.device, streams: List[Stream]):
        self.dataloader = dataloader
        self.buffer_size = buffer_size
        self.device = device
        self.streams = streams

    @staticmethod
    def thread_worker(prefetch_data: _MyPrefetchData):
        itr = iter(prefetch_data.dataloader)
        num_streams = len(prefetch_data.streams)
        while not prefetch_data.stop_iteration:
            # Run if not paused
            while prefetch_data.run_prefetcher:
                if (
                    len(prefetch_data.prefetch_buffer)
                    < prefetch_data.buffer_size
                ):
                    try:
                        item = next(itr)
                        current_stream = prefetch_data.streams[item.minibatch_idx % num_streams]
                        with torch.cuda.stream(current_stream):
                            item.to(prefetch_data.device, non_blocking=True)
                        prefetch_data.prefetch_buffer.append(item)
                    except Exception as e:  # pylint: disable=broad-except
                        prefetch_data.run_prefetcher = False
                        prefetch_data.stop_iteration = True
                        prefetch_data.prefetch_buffer.append(e)
                else:  # Buffer is full, waiting for main thread to consume items
                    # TODO: Calculate sleep interval based on previous consumption speed
                    time.sleep(PRODUCER_SLEEP_INTERVAL)
            prefetch_data.paused = True
            # Sleep longer when this prefetcher thread is paused
            time.sleep(PRODUCER_SLEEP_INTERVAL * 10)


    def __iter__(self):
        try:
            prefetch_data = _MyPrefetchData(
                self.dataloader,
                self.buffer_size,
                self.device,
                self.streams,
            )
            thread = threading.Thread(
                target=MyDataLoader.thread_worker,
                args=(prefetch_data,),
                daemon=True,
            )
            thread.start()

            while (
                not prefetch_data.stop_iteration
                or len(prefetch_data.prefetch_buffer) > 0
            ):
                if len(prefetch_data.prefetch_buffer) > 0:
                    data = prefetch_data.prefetch_buffer.popleft()
                    if isinstance(data, Exception):
                        if isinstance(data, StopIteration):
                            break
                        raise data
                    yield data
                else:
                    time.sleep(CONSUMER_SLEEP_INTERVAL)
        finally:
            if "prefetch_data" in locals():
                prefetch_data.run_prefetcher = False
                prefetch_data.stop_iteration = True
                prefetch_data.paused = False
            if "thread" in locals():
                thread.join()



class ItemSamplerFromFile(torch.utils.data.IterDataPipe):
    def __init__(
        self,
        item_set: gb.ItemSet,
        batch_size: int,
        root: Path,
    ) -> None:
        super().__init__()
        self._item_set = item_set
        self._names = item_set.names
        self._batch_size = batch_size
        self._root = root

    def __iter__(self) -> Iterator:
        total = len(self._item_set)
        output_count = total
        minibatch_idx = 0
        num_minibatches = (total + self._batch_size - 1) // self._batch_size
        reporter = TimingReporter('item_sampler_from_file', num_minibatches, 0)
        for i in range(0, total, self._batch_size):
            reporter.report_time('start', minibatch_idx)
            if output_count <= 0:
                break
            data = torch.load(self._root.joinpath(f'minibatch{minibatch_idx:04d}.pt'), weights_only=False)
            data = data.pin_memory()
            data.minibatch_idx = minibatch_idx # add a new attribute
            reporter.report_time('end', minibatch_idx)
            yield data
            output_count -= self._batch_size
            minibatch_idx += 1


def create_dataloader(
    graph, features, itemset, batch_size, fanout,
    storage_device, device, num_workers, job,
    reporter=None, no_copy=False, num_threads=None, num_minibatches=None,
    prefetch_factor=None, pre_samples_root=None,
    sample_mode=None, num_sampling_buffers=0,
):
    """
    [HIGHLIGHT]
    Get a GraphBolt version of a dataloader for node classification tasks.
    This function demonstrates how to utilize functional forms of datapipes in
    GraphBolt. For a more detailed tutorial, please read the examples in
    `dgl/notebooks/graphbolt/walkthrough.ipynb`.
    Alternatively, you can create a datapipe using its class constructor.

    Parameters
    ----------
    job : one of ["train", "evaluate", "infer"]
        The stage where dataloader is created, with options "train", "evaluate"
        and "infer".
    Other parameters are explicated in the comments below.
    """

    ############################################################################
    # [Step-1]:
    # gb.ItemSampler()
    # [Input]:
    # 'itemset': The current dataset. (e.g. `train_set` or `valid_set`)
    # 'batch_size': Specify the number of samples to be processed together,
    # referred to as a 'mini-batch'. (The term 'mini-batch' is used here to
    # indicate a subset of the entire dataset that is processed together. This
    # is in contrast to processing the entire dataset, known as a 'full batch'.)
    # 'job': Determines whether data should be shuffled. (Shuffling is
    # generally used only in training to improve model generalization. It's
    # not used in validation and testing as the focus there is to evaluate
    # performance rather than to learn from the data.)
    # [Output]:
    # An ItemSampler object for handling mini-batch sampling.
    # [Role]:
    # Initialize the ItemSampler to sample mini-batche from the dataset.
    ############################################################################
    if(pre_samples_root is None):
        datapipe = gb.ItemSampler(
            itemset, batch_size=batch_size, shuffle=(job == "train"),
            assign_minibatch_idx=True,
            num_threads=num_threads,
            pin_memory=num_workers == 0 and storage_device == "cpu"
        )
    else:
        datapipe = ItemSamplerFromFile(itemset, batch_size, pre_samples_root)

    ############################################################################
    # [Step-2]:
    # self.copy_to()
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device. Copying here
    # ensures that the rest of the operations run on the GPU.
    ############################################################################
    if storage_device != "cpu":
        datapipe = datapipe.copy_to(device=device)

    ############################################################################
    # [Step-3]:
    # self.sample_neighbor()
    # [Input]:
    # 'graph': The network topology for sampling.
    # '[-1] or fanout': Number of neighbors to sample per node. In
    # training or validation, the length of `fanout` should be equal to the
    # number of layers in the model. In inference, this parameter is set to
    # [-1], indicating that all neighbors of a node are sampled.
    # [Output]:
    # A NeighborSampler object to sample neighbors.
    # [Role]:
    # Initialize a neighbor sampler for sampling the neighborhoods of nodes.
    ############################################################################
    if(reporter is not None):
        datapipe = reporter.report_time('sampling', datapipe)
    if(pre_samples_root is None):
        datapipe = getattr(datapipe, 'sample_neighbor')(
            graph,
            fanout if job != "infer" else [-1],
            overlap_fetch=storage_device == "pinned",
            asynchronous=storage_device != "cpu",
            num_minibatches=num_minibatches,
            num_workers=num_workers,
            pin_memory=num_workers == 0 and storage_device == "cpu",
            return_picked_eids=False,
            sample_mode=sample_mode,
            num_sampling_buffers=num_sampling_buffers,
        )

    ############################################################################
    # [Step-4]:
    # self.fetch_feature()
    # [Input]:
    # 'features': The node features.
    # 'node_feature_keys': The keys of the node features to be fetched.
    # [Output]:
    # A FeatureFetcher object to fetch node features.
    # [Role]:
    # Initialize a feature fetcher for fetching features of the sampled
    # subgraphs.
    ############################################################################
    if(reporter is not None):
        datapipe = reporter.report_time('fetching', datapipe)
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"],
                                      overlap_fetch=True,
                                      user_set_num_stages=None,
                                      pin_memory_in_advance=num_workers > 0 and storage_device == "cpu")
    ############################################################################
    # [Step-5]:
    # self.copy_to()
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device.
    ############################################################################
    if(reporter is not None):
        datapipe = reporter.report_time('transfer', datapipe)
    if storage_device == "cpu" and not no_copy:
        datapipe = datapipe.copy_to(device=device)

    ############################################################################
    # [Step-6]:
    # gb.DataLoader()
    # [Input]:
    # 'datapipe': The datapipe object to be used for data loading.
    # 'num_workers': The number of processes to be used for data loading.
    # [Output]:
    # A DataLoader object to handle data loading.
    # [Role]:
    # Initialize a multi-process dataloader to load the data in parallel.
    ############################################################################
    if(reporter is not None):
        datapipe = reporter.report_time('end', datapipe)
    dataloader = gb.DataLoader(datapipe, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Return the fully-initialized DataLoader object.
    return dataloader
