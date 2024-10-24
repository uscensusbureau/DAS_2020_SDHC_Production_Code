import os
import sys
import time
import select

# colors
CRED = '\33[31m'
CWHITEBG2 = '\33[107m'
CEND = '\33[0m'

def das_tee(*, args):
    """Take input from STDIN and send it to the dashboard.
    """

    outfiles = []
    if args.stderr:
        outfiles.append(sys.stderr)
    else:
        outfiles.append(sys.stdout)
    if args.write:
        try:
            outfiles.append( open( args.write, 'w'))
        except FileNotFoundError as e:
            print(f"Cannot open {args.write} for writing",file=sys.stderr)
            print(f"Full path: {os.path.abspath(args.write)}", file=sys.stderr)
            print(f"Current directory: {os.getcwd()}",file=sys.stderr)
            raise e
    if args.red:
        sol = CRED+CWHITEBG2
        eol = CEND
    else:
        sol = ''
        eol = ''

    while True:
        linebuf = sys.stdin.readlines()
        lines = len(linebuf)
        if len(linebuf) == 0:
            exit(0)

        for out in outfiles:
            out.write(sol)
            for (ct,line) in enumerate(linebuf):
                out.write(f"{lines-len(linebuf)+ct:06d} {line}")
            out.write(eol)
            out.flush()

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description="writes output to stdout,stderr,or specified logfile")
    parser.add_argument('--red', action='store_true', help='When running tee, print output in red')
    parser.add_argument('--write', help='--tee also writes to this file')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--stdout', action='store_true', help='--tee writes to stdout (default)')
    group.add_argument('--stderr', action='store_true', help='--tee writes to stderr instead of stdout')
    group.add_argument("--errors", action='store_true', help="provide JSON error on STDIN message that is sent for the current mission to the dashboard")

    args = parser.parse_args()

    das_tee(args=args)

if __name__ == "__main__":
    main()