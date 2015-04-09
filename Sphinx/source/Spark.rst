
Spark on a local mahcine using 4 nodes
======================================

Started with

.. code:: bash

    MASTER=local[4] pyspark

Check that the SparkContext object is available.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    sc




.. parsed-literal::

    <pyspark.context.SparkContext at 0x10fbae510>



Example 1
~~~~~~~~~

Adapted from scala version in Chapter 2: Introduction to Data Analysis
with Scala and Spark of Advanced Analytics with Spark (O'Reilly 2015)

.. code:: python

    import os
    
    if not os.path.exists('documentation'):
        ! curl -o documentation https://archive.ics.uci.edu/ml/machine-learning-databases/00210/documentation
    if not os.path.exists('donation.zip'):
        ! curl -o donation.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00210/donation.zip
    ! unzip -n -q donation.zip
    ! unzip -n -q 'block_*.zip'
    if not os.path.exists('linkage'):
        ! mkdir linkage
    ! mv block_*.csv linkage
    ! rm block_*.zip


.. parsed-literal::

    
    10 archives were successfully processed.


.. code:: python

    ls


.. parsed-literal::

    EMR.pem          Untitled.ipynb   donation.zip     [31mword_count.py[m[m*
    EMR2.pem         [34mbooks[m[m/           frequencies.csv
    MapReduce.ipynb  [31mdocumentation[m[m*   [34mlinkage[m[m/


Info about the data set
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    ! cat documentation


.. parsed-literal::

    1. Title: Record Linkage Comparison Patterns 
    
    2. Source Information
       -- Underlying records: Epidemiologisches Krebsregister NRW
          (http://www.krebsregister.nrw.de)
       -- Creation of comparison patterns and gold standard classification:
          Institute for Medical Biostatistics, Epidemiology and Informatics (IMBEI),
          University Medical Center of Johannes Gutenberg University, Mainz, Germany
          (http://www.imbei.uni-mainz.de) 
       -- Donor: Murat Sariyar, Andreas Borg (IMBEI)    
       -- Date: September 2008
     
    3. Past Usage:
        1. Irene Schmidtmann, Gael Hammer, Murat Sariyar, Aslihan Gerhold-Ay:
           Evaluation des Krebsregisters NRW Schwerpunkt Record Linkage. Technical
           Report, IMBEI 2009. 
           http://www.krebsregister.nrw.de/fileadmin/user_upload/dokumente/Evaluation/EKR_NRW_Evaluation_Abschlussbericht_2009-06-11.pdf
           -- Describes the external evaluation of the registry's record linkage
              procedures.
           -- The comparison patterns in this data set were created in course of
              this evaluation.
               
        2. Murat Sariyar, Andreas Borg, Klaus Pommerening: 
           Controlling false match rates in record linkage using extreme value theory.
           Journal of Biomedical Informatics, 2011 (in press). 
           -- Predicted attribute: matching status (boolean).
           -- Results:
              -- A new approach for estimating the false match rate in record 
                 linkage by methods of Extreme Value Theory (EVT).
              -- The model eliminates the need for labelled training data while
                 achieving only slighter lower accuracy compared to a procedure
                 that has knowledge about the matching status.
    
    4. Relevant Information:
    
      The records represent individual data including first and 
      family name, sex, date of birth and postal code, which were collected through 
      iterative insertions in the course of several years. The comparison
    
      from 2005 to 2008. Data pairs were classified as "match" or "non-match" during 
      an extensive manual review where several documentarists were involved. 
      The resulting classification formed the basis for assessing the quality of the 
      registry’s own record linkage procedure.
      
      In order to limit the amount of patterns a blocking procedure was applied,
      which selects only record pairs that meet specific agreement conditions. The
      results of the following six blocking iterations were merged together:
      
        1. Phonetic equality of first name and family name, equality of date of birth.
        2. Phonetic equality of first name, equality of day of birth.
        3. Phonetic equality of first name, equality of month of birth.
        4. Phonetic equality of first name, equality of year of birth.
        5. Equality of complete date of birth.
        6. Phonetic equality of family name, equality of sex.
        
      This procedure resulted in 5.749.132 record pairs, of which 20.931 are matches.
      
      The data set is split into 10 blocks of (approximately) equal size and ratio
      of matches to non-matches.
    
      The separate file frequencies.csv contains for every predictive attribute 
      the average number of values in the underlying records. These values can, for example,
      be used as u-probabilities in weight-based record linkage following the
      framework of Fellegi and Sunter.
       
    
    5. Number of Instances: 5.749.132
    
    6. Number of Attributes: 12 (9 predictive attributes, 2 non-predictive, 
                                 1 goal field)
    
    7. Attribute Information:
       1. id_1: Internal identifier of first record.
       2. id_2: Internal identifier of second record.
       3. cmp_fname_c1: agreement of first name, first component
       4. cmp_fname_c2: agreement of first name, second component
       5. cmp_lname_c1: agreement of family name, first component
       6. cmp_lname_c2: agreement of family name, second component
       7. cmp_sex: agreement sex
       8. cmp_bd: agreement of date of birth, day component
       9. cmp_bm: agreement of date of birth, month component
       10. cmp_by: agreement of date of birth, year component
       11. cmp_plz: agreement of postal code
       12. is_match: matching status (TRUE for matches, FALSE for non-matches)
    
    8. Missing Attribute Values:  
    
      cmp_fname_c1: 1007
      cmp_fname_c2: 5645434
      cmp_lname_c1: 0
      cmp_lname_c2: 5746668
      cmp_sex:      0
      cmp_bd:       795
      cmp_bm:       795
      cmp_by:       795
      cmp_plz:      12843
    
    
    9. Class Distribution: 20.931 matches, 5728201 non-matches
    


If we are running Spark on Hadoop, we need to transfer files to HDFS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    ! hadoop fs -rm -rf linkage
    ! hadoop fs -put block_*.csv linkage

.. code:: python

    rdd = sc.textFile('linkage')

.. code:: python

    rdd.first()




.. parsed-literal::

    u'"id_1","id_2","cmp_fname_c1","cmp_fname_c2","cmp_lname_c1","cmp_lname_c2","cmp_sex","cmp_bd","cmp_bm","cmp_by","cmp_plz","is_match"'



