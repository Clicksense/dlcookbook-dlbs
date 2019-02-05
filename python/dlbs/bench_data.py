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
import argparse
import itertools
from six import string_types, integer_types

from dlbs.utils import IOUtils, OpenFile, DictUtils
from dlbs.processor import Processor


class DLPGUtils(object):

    EXPECTED_VALUES = {
        "exp.framework_title": ["TensorFlow", "Caffe", "Caffe2", "MXNET", "PyTorch", "TensorRT"],
        "exp.backend": ["caffe", "caffe2", "mxnet", "nvcnn", "nvtfcnn", "pytorch", "tensorflow", "tensorrt"],
        "exp.node_id": ["apollo_6500_xl_gen9", "apollo_6500_xl_gen10"],
        "exp.node_title": ["Apollo 6500 XL Gen9", "Apollo 6500 XL Gen10"],
        "exp.device_type": ["cpu", "gpu"],
        "exp.device_title": ["Tesla P100-PCIE-16GB", "Tesla P100-SXM2-16GB",
                             "Tesla P4", "Tesla T4",
                             "Tesla V100-PCIE-16GB", "Tesla V100-SXM2-16GB", "Tesla V100-SXM2-32GB"],
        "exp.num_node_gpus": [1, 2, 4, 8],
        "exp.dtype": ["float32", "float16", "int8"],
        "exp.data": ["synthetic", "real", "real/ssd", "real/dram", "real/weka", "real/nvme"],
        "exp.phase": ["training", "inference"],
        "exp.model_title": ["AcousticModel", "AlexNet", "AlexNetOWT", "DeepMNIST", "DeepSpeech2", "GoogleNet",
                            "InceptionV3", "InceptionV4", "Overfeat", "ResNet18", "ResNet34", "ResNet50",
                            "ResNet101", "ResNet152", "ResNet200", "ResNet269", "SensorNet", "Seq2SeqAutoencoder",
                            "TextCNN", "VGG11", "VGG13", "VGG16", "VGG19"]
    }

    @staticmethod
    def check_values(param, param_values, expected_values):
        unexpected_values = [val for val in param_values if val not in expected_values]
        if unexpected_values:
            print("Parameter '{}' has unexpected values {}. "
                  "Expecting one of {}.".format(param, unexpected_values, expected_values))
        else:
            print("Parameter '{}' check OK.".format(param))

    @staticmethod
    def check(bench_data):
        summary = bench_data.summary(params=list(DLPGUtils.EXPECTED_VALUES))
        for param in DLPGUtils.EXPECTED_VALUES:
            DLPGUtils.check_values(param, summary[param], DLPGUtils.EXPECTED_VALUES[param])


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
    def status(arg):
        """ Return status of the benchmark stored in a log file `log_file`.

        Args:
            arg: A name of a log file, a dictionary or an instance of the BenchData class.

        Returns:
            str or None: "ok" for successful benchmark, "failure" for not and None for other cases (such as no file).
        """
        if isinstance(arg, string_types):
            bench_data = BenchData.parse(arg)
        elif isinstance(arg, dict):
            bench_data = BenchData([arg], create_copy=False)
        elif isinstance(arg, BenchData):
            bench_data = arg
        else:
            raise TypeError("Invalid argument type (={}). Expecting string, BenchData".format(type(arg)))
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
        return BenchData(benchmarks['data'], create_copy=False)

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
        return BenchData(benchmarks, create_copy=False)

    def __init__(self, benchmarks=None, create_copy=False):
        if benchmarks is None:
            self.__benchmarks = []
        else:
            self.__benchmarks = copy.deepcopy(benchmarks) if create_copy else benchmarks

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
        return BenchData(selected, create_copy=False)

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
        return BenchData(benchmarks, create_copy=False)

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
        return BenchData(selected, create_copy=False)

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

    def report(self, inputs=None, output=None, output_cols=None, report_speedup=False, report_efficiency=False):
        reporter = BenchData.Reporter(self)
        reporter.report(inputs, output, output_cols, report_speedup, report_efficiency)

    class Reporter(object):
        TITLES = {
            "exp.model_title": "Model", "exp.replica_batch": "Replica Batch", "exp.effective_batch": "Effective Batch",
            "exp.num_gpus": "Num GPUs", "exp.gpus": "GPUs", "exp.dtype": "Precision",
            "exp.docker_image": "Docker Image"
        }

        @staticmethod
        def to_string(val):
            if val is None:
                return "-"
            elif isinstance(val, string_types):
                return val
            elif isinstance(val, integer_types):
                return "{:d}".format(val)
            elif isinstance(val, float):
                return "{:.2f}".format(val)
            else:
                raise TypeError("Invalid value type (='{}'). Expecting strings, integers or floats.".format(type(val)))

        def build_cache(self, inputs=None, output=None, output_cols=None):
            self.input_cols = [None] * len(inputs)
            for idx, param in enumerate(inputs):
                self.input_cols[idx] = {"index": idx, "param": param, "width": 0,
                                        "title": DictUtils.get(BenchData.Reporter.TITLES, param, param),
                                        "vals": sorted(self.bench_data.select_values(param))}
            self.output_param = output
            output_cols = output_cols if output_cols else sorted(self.bench_data.select_values(output))
            self.output_cols = [None] * len(output_cols)
            for idx, param_value in enumerate(output_cols):
                self.output_cols[idx] = {"index": idx, "value": param_value, "title": param_value,
                                         "width": len(BenchData.Reporter.to_string(param_value))}
            self.cache = {}
            for bench in self.bench_data.benchmarks():
                if BenchData.status(bench) != "ok":
                    continue
                bench_key = []
                for input_col in self.input_cols:
                    param_value = DictUtils.get(bench, input_col['param'], None)
                    if not param_value:
                        bench_key = []
                        break
                    bench_key.append(str(param_value))
                if bench_key:
                    output_val = DictUtils.get(bench, self.output_param, None)
                    if output_val:
                        bench_key = '.'.join(bench_key + [str(output_val)])
                        if bench_key not in self.cache:
                            self.cache[bench_key] = bench
                        else:
                            raise ValueError("Duplicate benchmark with key = {}".format(bench_key))

        def compute_column_widths(self, times, throughputs):
            # Input columns
            for input_col in self.input_cols:
                input_col['width'] = len(input_col['title'])
                for val in input_col['vals']:
                    input_col['width'] = max(input_col['width'], len(BenchData.Reporter.to_string(val)))
            # Output columns
            num_rows = len(times)
            num_output_cols = len(self.output_cols)
            for row_idx in range(num_rows):
                for col_idx in range(num_output_cols):
                    self.output_cols[col_idx]['width'] = max([
                        self.output_cols[col_idx]['width'],
                        len(BenchData.Reporter.to_string(times[row_idx][col_idx])),
                        len(BenchData.Reporter.to_string(throughputs[row_idx][col_idx]))
                    ])

        def compute_speedups(self, throughputs):
            speedups = copy.deepcopy(throughputs)
            num_cols = len(self.output_cols)
            for row in speedups:
                for idx in range(1, num_cols):
                    row[idx] = None if row[0] is None or row[idx] is None else float(row[idx]) / row[0]
                row[0] = 1.00 if row[0] is not None else None
            return speedups

        def compute_efficiency(self, times):
            replica_batch_idx = -1
            effective_batch_idx = -1
            for col_idx, input_col in enumerate(self.input_cols):
                if input_col['param'] == "exp.replica_batch":
                    replica_batch_idx = col_idx
                elif input_col['param'] == "exp.effective_batch":
                    effective_batch_idx = col_idx

            if (replica_batch_idx == -1 and effective_batch_idx == -1) or \
               (replica_batch_idx >= 0 and effective_batch_idx >= 0) or \
               self.output_param != "exp.num_gpus":
                raise ValueError("Efficiency can only be computed when one of the inputs is either replica or "
                                 "effective batch and when output is the number of GPUs e.g: "
                                 "inputs=['exp.model_title', 'exp.replica_batch'], output='exp.num_gpus'")
            efficiency = copy.deepcopy(times)
            num_cols = len(self.output_cols)
            for row in efficiency:
                for idx in range(1, num_cols):
                    if row[0] is None or row[idx] is None:
                        row[idx] = None
                    else:
                        if replica_batch_idx >= 0:
                            # Weak scaling
                            row[idx] = int(10000.0 * row[0] / row[idx]) / 100.0
                        else:
                            # String scaling
                            row[idx] = int(10000.0 * row[0] / (self.output_cols[idx]['value'] * row[idx])) / 100.0
                        row[idx] = min(row[idx], 100.0)
                row[0] = 100.00 if row[0] is not None else None
            return efficiency

        def get_header(self):
            header = ""
            for input_col in self.input_cols:
                format_str = "  %-" + str(input_col['width']) + "s"
                header = header + format_str % BenchData.Reporter.to_string(input_col['title'])
            header += "    "
            for output_col in self.output_cols:
                format_str = "%+" + str(output_col['width']) + "s  "
                header = header + format_str % BenchData.Reporter.to_string(output_col['title'])
            return header

        def print_table(self, title, header, inputs, outputs):
            print(title)
            print(header)
            for input, output in zip(inputs, outputs):
                row = ""
                for input_col in self.input_cols:
                    format_str = "  %-" + str(input_col['width']) + "s"
                    row = row + format_str % BenchData.Reporter.to_string(input[input_col['index']])
                row += "    "
                for output_col in self.output_cols:
                    format_str = "%+" + str(output_col['width']) + "s  "
                    row = row + format_str % BenchData.Reporter.to_string(output[output_col['index']])
                print(row)
            print("\n\n")

        def __init__(self, bench_data):
            self.bench_data = bench_data
            self.input_cols = None
            self.output_param = None
            self.output_cols = None
            self.cache = None

        def report(self, inputs=None, output=None, output_cols=None, report_speedup=False, report_efficiency=False):
            # Build cache that will map benchmarks keys to benchmark objects.
            self.build_cache(inputs, output, output_cols)
            # Iterate over column values and build table with batch times and throughput
            cols = []
            times = []
            throughputs = []
            benchmark_keys = [input_col['vals'] for input_col in self.input_cols]
            # Build tables for batch times and benchmarks throughputs
            # The `benchmark_key` is a tuple of column values e.g. ('ResNet50', 256)
            for benchmark_key in itertools.product(*benchmark_keys):
                cols.append(copy.deepcopy(benchmark_key))
                times.append([None] * len(self.output_cols))
                throughputs.append([None] * len(self.output_cols))
                for output_col in self.output_cols:
                    benchmark_key = [str(key) for key in benchmark_key]
                    key = '.'.join(benchmark_key + [str(output_col['value'])])
                    if key in self.cache:
                        times[-1][output_col['index']] = self.cache[key]['results.time']
                        throughputs[-1][output_col['index']] = self.cache[key]['results.throughput']
            # Determine minimal widths for columns
            self.compute_column_widths(times, throughputs)
            #
            header = self.get_header()
            self.print_table("Batch time (milliseconds)", header, cols, times)
            self.print_table("Throughput (instances per second e.g. images/sec)", header, cols, throughputs)
            if report_speedup:
                speedups = self.compute_speedups(throughputs)
                self.print_table("Speedup (based on instances per second table, "
                                 "relative to first output column ({} = {}))".format(self.output_param,
                                                                                     self.output_cols[0]['value']),
                                 header, cols, speedups)
            if report_efficiency:
                efficiency = self.compute_efficiency(times)
                self.print_table("Efficiency (based on batch times table, "
                                 "relative to first output column ({} = {}))".format(self.output_param,
                                                                                     self.output_cols[0]['value']),
                                 header, cols, efficiency)


def parse_arguments():
    """Parse command line arguments

    Returns:
        dict: Dictionary with command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='parse', choices=['parse', 'summary', 'report'],
                        help="Action to perform. ")
    parser.add_argument('inputs', type=str, nargs='*',
                        help="Input file(s) and/or folders. ")
    parser.add_argument('--no-recursive', '--no_recursive', required=False, default=False,
                        action='store_true', help='When parsing log files, do not parse folders recursively.')
    parser.add_argument('--select', type=str, required=False, default=None,
                        help="A select query to filter benchmarks.")
    parser.add_argument('--update', type=str, required=False, default=None,
                        help="An expression to update query benchmarks.")
    parser.add_argument('--output', type=str, required=False, default=None,
                        help="File to write output to. If not specified, standard output is used. When log parsing "
                             "is performed, several output formats are supported: '*.json' and '*.json.gz'.")
    return vars(parser.parse_args())


def parse(**kwargs):
    data = BenchData.parse(kwargs['inputs'], recursive=not kwargs['no-recursive'])
    if kwargs['select'] is not None:
        data = data.select(kwargs['select'])
    if kwargs['update'] is not None:
        data = data.update(kwargs['select'], use_processor=False)
    data.save(kwargs['output'])


def main():
    args = parse_arguments()
    print(args)
    if args['action'] == 'parse':
        parse(**args)


if __name__ == "__main__":
    main()
