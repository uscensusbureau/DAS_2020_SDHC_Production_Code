#!/usr/bin/env python3
"""Command line interface for PHSafe."""

# Copyright 2024 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
from argparse import ArgumentParser
from pathlib import Path

from tmlt.common.io_helpers import get_logger_stream, write_log_file
from tmlt.phsafe.runners import run_input_validation, run_tabulation


def main():
    """Parse command line arguments and run PHSafe."""
    parser = ArgumentParser(prog="ph-safe")
    subparsers = parser.add_subparsers(help="ph-safe sub-commands", dest="mode")

    def add_config(subparser: ArgumentParser) -> None:
        subparser.add_argument(
            dest="config_file", help="path to PHSafe config file", type=str
        )

    def add_data_path(subparser: ArgumentParser) -> None:
        subparser.add_argument(
            dest="data_path",
            help=(
                "string used by the reader. The string is interpreted as an "
                "input csv files directory path "
                "for a csv reader or as a reader config file path for a cef reader."
            ),
            type=str,
        )

    def add_output(subparser: ArgumentParser) -> None:
        subparser.add_argument(
            dest="output_path",
            help="name of directory that contains all output files",
            type=str,
        )

    def add_log(subparser: ArgumentParser) -> None:
        subparser.add_argument(
            "-l",
            "--log",
            dest="log_filename",
            help="path to log file",
            type=str,
            default="phsafe.log",
        )

    def add_validate_input(subparser: ArgumentParser) -> None:
        subparser.add_argument(
            "-v",
            "--validate",
            dest="input_validation_flag",
            help="validate inputs before running phsafe algorithm",
            action="store_true",
            default=False,
        )

    def add_validate_private_output(subparser: ArgumentParser) -> None:
        subparser.add_argument(
            "-vo",
            "--validate-private-output",
            dest="output_validation_flag",
            help="validate private outputs after running phsafe private algorithm",
            action="store_true",
            default=False,
        )

    add_log(parser)
    non_private_tab = subparsers.add_parser(
        name="non-private", help="non-private tabulations"
    )
    for add_arg_func in (
        add_config,
        add_data_path,
        add_output,
        add_log,
        add_validate_input,
    ):
        add_arg_func(non_private_tab)

    private_tab = subparsers.add_parser(name="private", help="private tabulations")
    for add_arg_func in (
        add_config,
        add_data_path,
        add_output,
        add_log,
        add_validate_input,
        add_validate_private_output,
    ):
        add_arg_func(private_tab)

    validate_tab = subparsers.add_parser(
        name="validate", help="validate inputs and config"
    )
    for add_arg_func in (add_config, add_data_path, add_log, add_validate_input):
        add_arg_func(validate_tab)

    args = parser.parse_args()

    # Set up logging.
    logger, io_stream = get_logger_stream()
    logger.info("PHSafe started.")

    if not args.mode:
        logger.error("No command was provided. Exiting...")
        sys.exit(1)

    if args.mode == "validate":
        logger.info("Validating PHSafe inputs and config ...")
        run_input_validation(config_path=args.config_file, data_path=args.data_path)
    else:
        if args.output_path:
            Path(args.output_path).mkdir(parents=True, exist_ok=True)
        logger.info("Running PHSafe in '%s' mode ...", args.mode)
        run_tabulation(
            config_path=args.config_file,
            data_path=args.data_path,
            output_path=args.output_path,
            should_validate_input=args.input_validation_flag,
            should_validate_private_output=(
                args.output_validation_flag if args.mode == "private" else False
            ),
            private=args.mode == "private",
        )
        logger.info("PHSafe finished.\n")

    if args.log_filename:
        log_content = io_stream.getvalue()
        io_stream.close()
        write_log_file(args.log_filename, log_content)


if __name__ == "__main__":
    main()
