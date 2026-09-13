import time
from torch.utils.data import get_worker_info


class TimingReporter:

    def __init__(self, operation, num_minibatches, num_workers):
        self.operation = operation
        self.num_minibatches = num_minibatches
        self.num_workers = num_workers if(num_workers > 0) else 1
        self.records = []

    def report_time(self, phase, minibatch_index):
        self.records.append((phase, minibatch_index, time.time_ns()))
        if(phase == 'end'
           and minibatch_index + self.num_workers >= self.num_minibatches):
            self._write_records()

    def _write_records(self):
        worker_info = get_worker_info()
        if(worker_info is None):
            filename = f'timestamp_{self.operation}.txt'
        else:
            filename = f'timestamp_{self.operation}{worker_info.id}.txt'

        with open(filename, 'wt') as f:
            for ph, idx, ts in self.records:
                f.write(f'{self.operation}({idx}) {ph}: {ts}\n')



class DataLoaderTimingReporter:

    def __init__(self, num_minibatches, num_workers):
        self.num_minibatches = num_minibatches
        self.num_workers = num_workers if(num_workers > 0) else 1
        self.records = []


    def report_time(self, operation, datapipe):
        if(operation == 'sampling'):
            return datapipe.transform(self._report_time_sampling_start)
        elif(operation == 'fetching'):
            return datapipe.transform(self._report_time_fetching_start)
        elif(operation == 'transfer'):
            return datapipe.transform(self._report_time_transfer_start)
        elif(operation == 'end'):
            return datapipe.transform(self._report_time_dataload_end)
    

    def _report_time_sampling_start(self, minibatch):
        return self._report_time('sampling', 'start', minibatch)

    #def _report_time_sampling_end(self, minibatch):
    #    return self._report_time('sampling', 'end', minibatch)

    def _report_time_fetching_start(self, minibatch):
        return self._report_time('fetching', 'start', minibatch)

    #def _report_time_fetching_end(self, minibatch):
    #    return self._report_time('fetching', 'end', minibatch)

    def _report_time_transfer_start(self, minibatch):
        return self._report_time('transfer', 'start', minibatch)

    def _report_time_dataload_end(self, minibatch):
        return self._report_time('dataload', 'end', minibatch)


    def _write_records(self):
        worker_info = get_worker_info()
        if(worker_info is None):
            filename = 'timestamp_dataloader.txt'
        else:
            filename = f'timestamp_dataloader{worker_info.id}.txt'

        with open(filename, 'wt') as f:
            for op, ph, idx, ts in self.records:
                f.write(f'{op}({idx}) {ph}: {ts}\n')

    def _report_time(self, operation, phase, minibatch):
        self.records.append((operation, phase, minibatch.minibatch_idx, time.time_ns()))
        if((operation == 'fetching' or operation == 'dataload')
           and minibatch.minibatch_idx + self.num_workers >= self.num_minibatches):
            self._write_records()
        return minibatch
