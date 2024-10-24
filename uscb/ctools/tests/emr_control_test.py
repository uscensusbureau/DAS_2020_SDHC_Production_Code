#!/usr/bin/env python

import os
import ctools.emr_control as emr_control
import pytest

@pytest.mark.skip(reason="Custom AMI works without this setup")
def test_user_in_group():
    assert emr_control.user_in_group('yarn',       'hadoop') == True
    assert emr_control.user_in_group('mapred',     'hadoop') == True
    assert emr_control.user_in_group('hdfs',       'hadoop') == True
    assert emr_control.user_in_group('kms',        'hadoop') == True
    assert emr_control.user_in_group('nosuchuser', 'hadoop') == False
    assert emr_control.user_in_group('yarn',       'nosuchgroup') == False

def test_requestInstanceCounts():
    core_test = 0
    task_test = 0
    for ig in emr_control.getInstanceGroups():
        if "core" in ig['Name'].lower():
            core_test  = ig[emr_control.RUNNING_INSTANCE_COUNT]+1
        if "task" in ig['Name'].lower():
            task_test = ig[emr_control.RUNNING_INSTANCE_COUNT]+1
    assert emr_control.requestInstanceCounts(core_test, task_test, dry_run="true")

if __name__=="__main__":
    test_user_in_group()
