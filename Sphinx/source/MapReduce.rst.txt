
Hadoop MapReduce on AWS EMR with ``mrjob``
==========================================

`Elastic MapReduce
Quickstart <https://pythonhosted.org/mrjob/guides/emr-quickstart.html>`__

MapReduce code
--------------

.. code:: python

    %%file word_count.py
    # From http://mrjob.readthedocs.org/en/latest/guides/quickstart.html#writing-your-first-job
    from mrjob.job import MRJob
    
    class MRWordFrequencyCount(MRJob):
    
        def mapper(self, _, line):
            yield "chars", len(line)
            yield "words", len(line.split())
            yield "lines", 1
    
        def reducer(self, key, values):
            yield key, sum(values)
    
    
    if __name__ == '__main__':
        MRWordFrequencyCount.run()


.. parsed-literal::

    Overwriting word_count.py


Configuration file
------------------

.. code:: python

    %%file ~/.mrjob.conf
    
    runners:
      emr:
        aws_access_key_id: <Your AWS access key>
        aws_secret_access_key: <Your AWS secret key>
        ec2_key_pair: <Your key_pair name>
        ec2_key_pair_file: <Location of PEM file>
        ssh_tunnel_to_job_tracker: true
        ec2_master_instance_type: c3.xlarge
        ec2_instance_type: c3.xlarge
        num_ec2_instances: 3


.. parsed-literal::

    Overwriting /Users/cliburn/.mrjob.conf


Launching job
-------------

.. code:: python

    %%bash
    
    python word_count.py -r emr s3://cliburn-sta663/books/*txt \
        --output-dir=s3://cliburn-sta663/wc_out \
        --no-output

Notes
     

Due to a recent change in Amazon policy, this won't work on accounts
created after 6 April 2015 due to the need to provide IAM roles. Until
``mrjob`` is updated to support this, the launch will fail with an
error.

