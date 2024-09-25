# Script for creating a packaged repository `repo.zip` to be uploaded to s3 bucket for spark cluster mode deployment.

# This script has dependency on associative arrays and requires bash version 4+.

usage() {
    echo "Usage: $0 -t <path to phsafe repository> [-r <path to CEF reader] " 1>&2; exit 1;
}

while getopts t:r: flag
do
    case "$flag" in
        t)
            staging_repo=$OPTARG
            echo "Creating repo.zip from $staging_repo"
            ;;
        r)
            cef_reader_path=$OPTARG
            ;;
        *)
            echo "Invalid argument: $flag"
            usage
            exit 1
            ;;
    esac
done

repo_zip=$(readlink -f $staging_repo)/repo.zip
for m in common core analytics phsafe; do
    pushd $staging_repo/$m
    zip -r $repo_zip tmlt
    popd
done

# If CEF reader is included, add it and remove the mock CEF reader.
if [ ! -z "$cef_reader_path" ]; then
    echo "Adding CEF reader from $cef_reader_path"

    pushd $cef_reader_path
    zip -r $repo_zip *
    popd
fi

pushd $staging_repo
touch "__init__.py"
zip $repo_zip __init__.py
rm "__init__.py"
popd
