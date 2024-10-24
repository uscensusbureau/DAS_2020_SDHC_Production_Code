# return the AWS errors as a JSON object

import os
import subprocess
import json
import ctools.aws as aws
import dashboard_client.dashboard_client as dashboard


def collect(args):
    DAS_S3 = os.getenv('DAS_S3ROOT').replace('s3://', '')

    errors = {}
    # not collecting 'AllRequests','GetRequests'
    for name in ['4xxErrors', '5xxErrors']:
        with aws.Proxy():
            cmd = ['aws', 'cloudwatch', 'get-metric-statistics', '--namespace', 'AWS/S3', '--metric-name', name,
                   '--start-time', args.start, '--end-time', args.end, '--period', '3600',
                   '--statistics', 'Sum',
                   '--unit', 'Count',
                   '--output', 'json',
                   '--dimensions', f'Name=BucketName,Value={DAS_S3}',
                   'Name=FilterId,Value=EntireBucket']
            err = json.loads(subprocess.check_output(cmd))
            if len(err['Datapoints']) == 0:
                continue
            errors[name] = err
    if not errors:
        print("No S3 errors")
    else:
        print(json.dumps(errors, indent=4, default=str))
        obj = {'errors': json.dumps(errors, default=str),
               'debug': args.debug}
        if args.upload:
            dashboard.send_obj(obj=obj, sender=dashboard.DASLOG_SENDER)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description="return AWS S3 errors between two times as a JSON object")
    parser.add_argument("--upload", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("start")
    parser.add_argument("end")
    args = parser.parse_args()

    collect(args)


if __name__ == "__main__":
    main()
