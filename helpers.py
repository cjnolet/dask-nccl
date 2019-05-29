from tornado import gen
from dask.distributed import default_client
from toolz import first
import logging
import dask.dataframe as dd

from dask.distributed import wait


def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def build_host_dict(workers):
    hosts = set(map(lambda x: parse_host_port(x), workers))
    hosts_dict = {}
    for host, port in hosts:
        if host not in hosts_dict:
            hosts_dict[host] = set([port])
        else:
            hosts_dict[host].add(port)

    return hosts_dict

def _build_host_dict(gpu_futures, client):
    """
    Helper function to build a dictionary mapping workers to parts
    that currently hold the parts of given futures.
    :param gpu_futures:
    :param client:
    :return:
    """
    who_has = client.who_has(gpu_futures)

    key_to_host_dict = {}
    for key in who_has:
        key_to_host_dict[key] = parse_host_port(who_has[key][0])

    hosts_to_key_dict = {}
    for key, host in key_to_host_dict.items():
        if host not in hosts_to_key_dict:
            hosts_to_key_dict[host] = set([key])
        else:
            hosts_to_key_dict[host].add(key)

    workers = [key[0] for key in list(who_has.values())]
    return build_host_dict(workers)

def to_gpu_matrix(i):
    return i

@gen.coroutine
def _get_mg_info(ddf):
    """
    Given a Dask cuDF, extract number of dimensions and convert
    the pieces of the Dask cuDF into Numba arrays, which can
    be passed into the kNN algorithm.
    build a
    :param ddf:
    :return:
    """

    client = default_client()

    if isinstance(ddf, dd.DataFrame):
        cols = len(ddf.columns)
        parts = ddf.to_delayed()
        parts = client.compute(parts)
        yield wait(parts)
    else:
        raise Exception("Input should be a Dask DataFrame")

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = [(worker, client.submit(to_gpu_matrix, part,
                                       workers=[worker]))
                for worker, part in worker_map]

    yield wait(gpu_data)

    raise gen.Return((gpu_data, cols))