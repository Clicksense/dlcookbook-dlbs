# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LogParser:
    $ python bench_data.py parse inputs --recursive --output [FILENAME]
BenchStats:
    $ python bench_data.py summary [FILENAME] --select SELECT --update UPDATE
SummaryBuilder
    $ python bench_data.py report  [FILENAME] --select SELECT --update UPDATE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
from six import string_types

from dlbs.utils import IOUtils, OpenFile, DictUtils
from dlbs.processor import Processor


def print_vals(obj):
    """A helper to print JSON with predefined indent. Is widely used in python notebooks.

    Args:
        obj: Something to print with json.dumps.
    """
    print(json.dumps(obj, indent=2))


class BenchData(object):

    @staticmethod
    def get_selector(query):
        """Returns a callable object that returns true when `query` matches dictionary.

        Args:
            query: An object that specifies the query. It can be one of the following:
                - A string:
                    - Load JSON object from this string if possible ELSE
                    - Treat it as a file name and load JSON objects from there.
                  The parsed/loaded object must be either dict or list.
                - A list of dict. Wrap it into a function that calls match method of a DictUtils class.
                - Callable object. Return as is.

        Returns:
            Callable object.
        """
        # If it's a string, assume it's a JSON parsable string and if not - assume it's a JSON file name.
        if isinstance(query, string_types):
            try:
                query = json.loads(query)
            except ValueError:
                query = IOUtils.read_json(query)

        selector = query
        # If it's a list of dict, wrap it into a function.
        if isinstance(query, (list, dict)):
            def dict_matcher(bench): return DictUtils.match(bench, query, policy='strict')
            selector = dict_matcher
        # Here, it must be a callable object.
        if not callable(selector):
            raise ValueError("Invalid type of object that holds parameters (%s)" % type(selector))
        return selector

    @staticmethod
    def status(log_file):
        """ Return status of the benchmark stored in a log file `log_file`.

        Args:
            log_file (str): A name of a log file.

        Returns:
            str or None: "ok" for successful benchmark, "failure" for not and None for other cases (such as no file).
        """
        bench_data = BenchData.parse(log_file)
        if len(bench_data) == 1:
            return 'ok' if DictUtils.get(bench_data[0], 'results.time', -1) > 0 else 'failure'
        return None

    @staticmethod
    def load(file_name):
        """Load benchmark data (parsed from log files) from a JSON file.

        A file name is a JSON file that contains object with 'data' field. This field
        is a list with dictionaries, each dictionary contains parameters for one benchmark:
        {"data":[{...}, {...}, {...}]}

        Args:
            file_name (str): File name of a JSON (*.json) or a compressed JSON (.json.gz) file.

        Returns:
            Instance of this class.
        """
        benchmarks = IOUtils.read_json(file_name, check_extension=True)
        if 'data' not in benchmarks:
            raise ValueError("No benchmark data found in '{}'".format(file_name))
        return BenchData(benchmarks['data'])

    @staticmethod
    def parse(inputs, recursive=False):
        """Parse benchmark log files (*.log).

        Args:
            inputs: Path specifiers of where to search for log files.
            recursive (bool): If true, parse directories found in `inputs` recursively.

        Returns:
            Instance of this class.
        """
        file_names = IOUtils.gather_files(inputs, "*.log", recursive)
        benchmarks = []
        for file_name in file_names:
            parameters = {}
            with OpenFile(file_name, 'r') as logfile:
                # The 'must_match' must be set to false. It says that not
                # every line in a log file must match key-value pattern.
                DictUtils.add(
                    parameters,
                    logfile,
                    pattern='[ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)',
                    must_match=False
                )
            benchmarks.append(parameters)
        return BenchData(benchmarks)

    def __init__(self, benchmarks=None):
        self.__benchmarks = copy.deepcopy(benchmarks) if benchmarks is not None else []

    def __len__(self):
        """Return number of benchmarks.

        Returns:
            Number of benchmarks.
        """
        return len(self.__benchmarks)

    def __getitem__(self, i):
        """Return parameters for the i-th benchmark

        Args:
            i (int): Benchmark index.

        Returns:
            dict: Parameters for i-th benchmark.
        """
        return self.__benchmarks[i]

    def benchmarks(self):
        """Return list of dictionaries with benchmark parameters.

        Returns:
            List of dictionaries where each dictionary contains parameters for one benchmarks.
        """
        return self.__benchmarks

    def clear(self):
        """Remove all benchmarks."""
        self.__benchmarks = []

    def copy(self):
        """Create a copy of this bench data instance.

        Returns:
            Copy of this instance.
        """
        return BenchData(copy.deepcopy(self.__benchmarks))

    def save(self, file_name):
        """ Save contents of this instance into a (compressed) JSON file.

        Args:
            file_name (str): A file name.
        """
        IOUtils.write_json(file_name, {'data': self.__benchmarks})

    def select(self, query):
        """ Select only those benchmarks that match `query`

        Args:
            query: Anything that's a valid with respect to `get_selector` method.

        Returns:
            BenchData: That contains those benchmarks that match `query`.
        """
        match = BenchData.get_selector(query)
        selected = [bench for bench in self.__benchmarks if match(bench)]
        return BenchData(selected)

    def delete(self, query):
        """ Delete only those benchmarks that match `query`

        Args:
            query: Anything that's a valid with respect to `get_selector` method.

        Returns:
            BenchData: That contains those benchmarks that do not match `query`.
        """
        match = BenchData.get_selector(query)
        return self.select(lambda bench: not match(bench))

    def update(self, query, use_processor=False):
        """Update benchmarks returning updated copy.

        Args:
            query: dict or callable.
            use_processor (bool): If true, apply variable processor. Will silently produce wrong results if
                benchmarks contain values that are dicts or lists.

        Returns:
            BenchData: Updated copy of benchmarks.
        """
        update_fn = query
        if isinstance(query, dict):
            def dict_update_fn(bench): bench.update(query)
            update_fn = dict_update_fn
        if not callable(update_fn):
            raise ValueError("Invalid update object (type='%s'). Expecting callable." % type(update_fn))

        benchmarks = copy.deepcopy(self.__benchmarks)
        for benchmark in benchmarks:
            update_fn(benchmark)

        if use_processor:
            Processor().compute_variables(benchmarks)
        return BenchData(benchmarks)

    def select_keys(self, keys):
        """Return copy of benchmarks that only contain `keys`

        Args:
            keys (list): List of benchmark keys to keep

        Returns:
            BenchData: Copy of current benchmarks with parameters defined in `keys`.
        """
        if keys is None:
            return self.copy()
        selected = [copy.deepcopy(DictUtils.subdict(bench, keys)) for bench in self.__benchmarks]
        return BenchData(selected)

    def select_values(self, key):
        """Return unique values for the `key` across all benchmarks.

        A missing key in a benchmark is considered to be a key having None value.

        Args:
            key (str): A key to return unique values for.

        Returns:
            list: sorted list of values.
        """
        selected = set()
        for benchmark in self.__benchmarks:
            selected.add(DictUtils.get(benchmark, key, None))
        return sorted(list(selected))

    def summary(self, params=None):
        """Return summary of benchmarks providing additional info on `params`.

        Args:
            params (list): List of parameters to provide additional info for. If empty, default list is used.

        Returns:
            dict: A summary of benchmarks.
        """
        if not params:
            params = ['exp.node_id', 'exp.node_title', 'exp.gpu_title', 'exp.gpu_id', 'exp.framework_title',
                      'exp.framework_id']
        summary_dict = {
            'num_benchmarks': len(self.__benchmarks),
            'num_failed_benchmarks': 0,
            'num_successful_benchmarks': 0
        }
        for param in params:
            summary_dict[param] = set()

        for bench in self.__benchmarks:
            if DictUtils.get(bench, 'results.time', -1) > 0:
                summary_dict['num_successful_benchmarks'] += 1
            else:
                summary_dict['num_failed_benchmarks'] += 1
            for param in params:
                summary_dict[param].add(DictUtils.get(bench, param, None))

        for param in params:
            summary_dict[param] = list(summary_dict[param])
        return summary_dict

    def report(self, report_type='exploration'):
        """Print exploration, weak/strong scaling reports.

        Args:
            report_type (str): Type of a report to build. One of 'exploration', 'weak-scaling' or 'strong-scaling'.
        """
        if report_type not in ['exploration', 'weak-scaling', 'strong-scaling']:
            raise ValueError("Invalid report type")
        benchmarks = self.select(self.Reporter.PARAMETERS)
        if len(benchmarks) == 0:
            raise("WARNING - reporter needs the following parameters "
                  "to be present in each benchmarks: {}".format(self.Reporter.PARAMETERS))
        reporter = BenchData.Reporter(benchmarks.benchmarks())
        if report_type == 'exploration':
            reporter.exploration_report()
        else:
            reporter.scaling_report(BenchData.Reporter.WEAK_SCALING if report_type == 'weak-scaling' else BenchData.Reporter.STRONG_SCALING)

    class Reporter(object):

        PARAMETERS = ['results.time', 'exp.model_title', 'exp.gpus', 'exp.effective_batch']
        BATCH_TM_TITLE = "Batch time (milliseconds)"
        IPS_TITLE = "Inferences Per Second (IPS, throughput)"
        SPEEDUP_TITLE = "Speedup (instances per second)"
        WEAK_SCALING = 1
        STRONG_SCALING = 2

        def __init__(self, benchmarks):
            self.cache = {}
            self.nets = set()
            self.batches = set()
            self.devices = set()
            for benchmark in benchmarks:
                key = '{0}_{1}_{2}'.format(
                    benchmark['exp.model_title'],
                    benchmark['exp.gpus'],
                    benchmark['exp.effective_batch']
                )
                if key in self.cache:
                    raise ValueError("Duplicate benchmark found with key={}".format(key))
                self.cache[key] = float(benchmark['results.time'])
                self.nets.add(benchmark['exp.model_title'])
                self.batches.add(int(benchmark['exp.effective_batch']))
                self.devices.add(str(benchmark['exp.gpus']))
            self.nets = sorted(list(self.nets))
            self.batches = sorted(list(self.batches))
            self.devices = sorted(list(self.devices), key=len)

        def exploration_report(self):
            """ Builds exploration report for inference and single device training.
            """
            header = "%-20s %-10s" % ('Network', 'Device')
            for batch in self.batches:
                header = "%s %-10s" % (header, batch)
            report = []
            num_batches = len(self.batches)
            for net in self.nets:
                for device in self.devices:
                    profile = {'net': net, 'device': device, 'time': [-1]*num_batches, 'throughput': [-1]*num_batches}
                    profile_ok = False
                    for idx, batch in enumerate(self.batches):
                        key = '{0}_{1}_{2}'.format(net, device, batch)
                        if key in self.cache and self.cache[key] > 0:
                            profile_ok = True
                            profile['time'][idx] = self.cache[key]
                            profile['throughput'][idx] = int(batch * (1000.0 / self.cache[key]))
                    if profile_ok:
                        report.append(profile)
            BenchData.Reporter.print_report_txt(BenchData.Reporter.BATCH_TM_TITLE, header, report, 'net', 'device', 'time')
            BenchData.Reporter.print_report_txt(BenchData.Reporter.IPS_TITLE, header, report, 'net', 'device', 'throughput')

        def scaling_report(self, report_type):
            """ Builds weak/strong scaling report for multi-GPU training.
            """
            header = "%-20s %-10s" % ('Network', 'Batch')
            for device in self.devices:
                header = "%s %-10d" % (header, (1 + device.count(',')))
            report = []
            for net in self.nets:
                for batch in self.batches:
                    profile = {'net': net, 'batch': batch, 'time': [], 'throughput': [], 'efficiency': [], 'speedup': []}
                    profile_ok = False
                    for device in self.devices:
                        num_devices = 1 + device.count(',')
                        effective_batch = batch
                        if report_type == BenchData.Reporter.WEAK_SCALING:
                            effective_batch = effective_batch * num_devices
                        key = '{0}_{1}_{2}'.format(net, device, effective_batch)
                        if num_devices == 1 and key not in self.cache:
                            # If we do not have data for one device, does not make sense to continue
                            break
                        batch_tm = throughput = efficiency = speedup = -1.0
                        if key in self.cache:
                            batch_tm = self.cache[key]
                            throughput = int(effective_batch * (1000.0 / batch_tm))
                            if len(profile['throughput']) == 0:
                                speedup = 1
                            else:
                                speedup = 1.0 * throughput / profile['throughput'][0]
                        if len(profile['efficiency']) == 0:
                            efficiency = 100.00
                            profile_ok = True
                        elif profile['time'][0] > 0:
                            if report_type == BenchData.Reporter.WEAK_SCALING:
                                efficiency = int(10000.0 * profile['time'][0] / batch_tm) / 100.0
                            else:
                                efficiency = int(10000.0 * profile['time'][0] / (num_devices * batch_tm)) / 100.0
                            profile_ok = True
                        profile['time'].append(batch_tm)
                        profile['throughput'].append(int(throughput))
                        profile['efficiency'].append(efficiency)
                        profile['speedup'].append(speedup)
                    if profile_ok:
                        report.append(profile)
            BenchData.Reporter.print_report_txt(BenchData.Reporter.BATCH_TM_TITLE, header, report, 'net', 'batch', 'time')
            BenchData.Reporter.print_report_txt(BenchData.Reporter.IPS_TITLE, header, report, 'net', 'batch', 'throughput')
            BenchData.Reporter.print_report_txt(BenchData.Reporter.SPEEDUP_TITLE, header, report, 'net', 'batch', 'speedup')
            BenchData.Reporter.print_report_txt(
                "Efficiency  = 100% * t1 / tN",
                header, report, 'net', 'batch', 'efficiency'
            )

        @staticmethod
        def print_report_txt(description, header, report, col1_key, col2_key, data_key):
            """ Writes a human readable report to a standard output.
            """
            print(description)
            print(header)
            for record in report:
                row = "%-20s %-10s" % (record[col1_key], record[col2_key])
                for idx in range(len(record['time'])):
                    val = record[data_key][idx]
                    if val >= 0:
                        if isinstance(val, int):
                            row = "%s %-10d" % (row, record[data_key][idx])
                        else:
                            row = "%s %-10.2f" % (row, record[data_key][idx])
                    else:
                        row = "%s %-10s" % (row, '-')
                print(row)
            print("\n\n")


def main():
    pass


if __name__ == "__main__":
    main()
