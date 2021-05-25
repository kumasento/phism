#!/usr/bin/env python3
# Summarize the results from pb-flow runs.

import os
import argparse
import glob
import pprint

import python.utils.polybench as pb_utils


def main():
    pp = pprint.PrettyPrinter(indent=2)

    parser = argparse.ArgumentParser(
        description='Process pb-flow run results.')
    parser.add_argument('result_dir', type=str,
                        help='pb-flow result directory.')
    args = parser.parse_args()

    print('\n>>> pb-flow report CLI\n')
    print('    Result directory: {}'.format(args.result_dir))
    print('\n')

    records = pb_utils.process_pb_flow_result_dir(args.result_dir)

    print('\n>>> Parsed results:')
    pp.pprint(records)


if __name__ == '__main__':
    main()