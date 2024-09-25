#! /bin/bash -xe

if [ $# -eq 0 ]; then
    echo "s3 wheel path must be supplied as an argument to this script"
    exit 2
fi

aws s3 cp $1 .
sudo $(which python3.11) -m pip install --no-deps $(basename $1)
