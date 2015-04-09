
Getting started with Python and the IPython notebook
====================================================

The IPython notebook is an interactive, web-based environment that
allows one to combine code, text and graphics into one unified document.
All of the lectures in this course have been developed using this tool.
In this lecture, we will introduce the notebook interface and
demonstrate some of its features.

**New**: A new version of the IPython notebook knowan as Jupyter
supports multiple kernels (differnet languages) and other enhancements.
For a tour of its features, see this
`notebook <http://nbviewer.ipython.org/urls/bitbucket.org/ipre/calico/raw/master/notebooks/Documentation/Reference%20Guide/Reference%20Guide.ipynb>`__.

Cells
-----

The IPython notebook has two types of cells:

::

    * Markdown
    * Code

Markdown is for text, and even allows some typesetting of mathematics,
while the code cells allow for coding in Python and access to many other
packages, compilers, etc.

Markdown
~~~~~~~~

To enter a markdown cell, just choose the cell tab and set the type to
'Markdown'.

.. code:: python

    from IPython.display import Image

.. code:: python

    Image(filename='screenshot.png')




.. image:: IPythonNotebookIntroduction_files/IPythonNotebookIntroduction_8_0.png



The current cell is now in Markdown mode, and whatever is entered is
assumed to be markdown code. For example, text can be put into *italics*
or **bold**. A bulleted list can be entered as follows:

Bulleted List \* Item 1 \* Item 2

Markdown has many features, and a good reference is located at:

http://daringfireball.net/projects/markdown/syntax

Code Cells
----------

Code cells take Python syntax as input. We will see a lot of those
shortly, when we begin our introduction to Python. For the moment, we
will highlight additional uses for code cells.

Magic Commands
--------------

Magic commands work a lot like OS command line calls - and in fact, some
are just that. To get a list of available magics:

.. code:: python

    %lsmagic




.. parsed-literal::

    Available line magics:
    %alias  %alias_magic  %autocall  %automagic  %autosave  %bookmark  %cat  %cd  %clear  %colors  %config  %connect_info  %cp  %debug  %dhist  %dirs  %doctest_mode  %ed  %edit  %env  %gui  %hist  %history  %install_default_config  %install_ext  %install_profiles  %killbgscripts  %ldir  %less  %lf  %lk  %ll  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %lx  %macro  %magic  %man  %matplotlib  %mkdir  %more  %mv  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %popd  %pprint  %precision  %profile  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %rep  %rerun  %reset  %reset_selective  %rm  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%debug  %%file  %%html  %%javascript  %%latex  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.



Notice there are line and cell magics. Line magics take the entire line
as argument, while cell magics take the cell. As 'automagic' is on, we
can omit the % when making calls to line magics.

.. code:: python

    ls



.. parsed-literal::

    AlgorithmicComplexity.ipynb
    AlgorithmicComplexity.rst
    [34mAlgorithmicComplexity_files[m[m/
    Animation.ipynb
    Animation.rst
    [34mAnimation_files[m[m/
    BlackBoxOptimization.ipynb
    BlackBoxOptimization.rst
    [34mBlackBoxOptimization_files[m[m/
    CUDAPython.ipynb
    CUDAPython.rst
    [34mCUDAPython_files[m[m/
    CalibratingODEs.ipynb
    CalibratingODEs.rst
    [34mCalibratingODEs_files[m[m/
    ComputationalStatisticsMotivation.ipynb
    ComputationalStatisticsMotivation.rst
    [34mComputationalStatisticsMotivation_files[m[m/
    ComputerArithmetic.ipynb
    ComputerArithmetic.rst
    [34mComputerArithmetic_files[m[m/
    CrashCourseInC.ipynb
    CrashCourseInC.rst
    DataProcessing-Solutions.ipynb
    DataProcessing-Solutions.rst
    [34mDataProcessing-Solutions_files[m[m/
    DataProcessing.ipynb
    DataProcessing.rst
    [34mDataProcessing_files[m[m/
    DistributedComputing.ipynb
    DistributedComputing.rst
    EM_algorithm.ipynb
    EM_algorithm.rst
    [34mEM_algorithm_files[m[m/
    FromCToPython.ipynb
    FromCToPython.rst
    FromCompiledToPython.ipynb
    FromCompiledToPython.rst
    FromJuliaToPython.ipynb
    FromJuliaToPython.rst
    [34mFromJuliaToPython_files[m[m/
    FromPythonToC.ipynb
    FromPythonToC.rst
    Functions-Solutions.rst
    Functions.ipynb
    Functions.rst
    FunctionsSolutions.ipynb
    FunctionsSolutions.rst
    GPUSAndCUDAC.rst
    [34mGPUs and Cuda C_files[m[m/
    GPUsAndCUDAC.ipynb
    [34mGPUsAndCUDAC_files[m[m/
    IP2.rst
    IPythonNotebookIntroduction.ipynb
    IPythonNotebookIntroduction.rst
    [34mIPythonNotebookIntroduction_files[m[m/
    IPythonNotebookPolyglot.ipynb
    IPythonNotebookPolyglot.rst
    [34mIPythonNotebookPolyglot_files[m[m/
    IntroductionToPython-Solutions.ipynb
    IntroductionToPython-Solutions.rst
    IntroductionToPython.ipynb
    IntroductionToPython.rst
    LinearAlgebraMatrixDecomp-WithSoutions.ipynb
    LinearAlgebraMatrixDecomp-WithSoutions.rst
    LinearAlgebraMatrixDecomp.ipynb
    LinearAlgebraMatrixDecomp.rst
    LinearAlgebraReview.ipynb
    LinearAlgebraReview.rst
    [34mLinearAlgebraReview_files[m[m/
    MCMC.ipynb
    MCMC.rst
    [34mMCMC_files[m[m/
    Makefile
    Makefile~
    MakingCodeFast.ipynb
    MakingCodeFast.rst
    [34mMakingCodeFast_files[m[m/
    MapReduce.ipynb
    MapReduce.rst
    MonteCarlo.ipynb
    MonteCarlo.rst
    [34mMonteCarlo_files[m[m/
    MultivariateOptimizationAlgortihms.ipynb
    MultivariateOptimizationAlgortihms.rst
    OptimizationInOneDimension.ipynb
    OptimizationInOneDimension.rst
    [34mOptimizationInOneDimension_files[m[m/
    Optimization_Bakeoff.ipynb
    Optimization_Bakeoff.rst
    PCA-Solutions.ipynb
    PCA-Solutions.rst
    [34mPCA-Solutions_files[m[m/
    PCA.ipynb
    PCA.rst
    [34mPCA_files[m[m/
    PyMC2.ipynb
    PyMC2.rst
    [34mPyMC2_files[m[m/
    PyMC3.ipynb
    PyMC3.rst
    [34mPyMC3_files[m[m/
    PyStan.ipynb
    PyStan.rst
    [34mPyStan_files[m[m/
    ResamplingAndMonteCarloSimulations.ipynb
    ResamplingAndMonteCarloSimulations.rst
    [34mResamplingAndMonteCarloSimulations_files[m[m/
    Spark.ipynb
    Spark.rst
    TextProcessing-Solutions.ipynb
    TextProcessing-Solutions.rst
    TextProcessing.ipynb
    TextProcessing.rst
    TextProcessingExtras.ipynb
    TextProcessingExtras.rst
    UsingNumpy-Solutions.ipynb
    UsingNumpy-Solutions.rst
    [34mUsingNumpy-Solutions_files[m[m/
    UsingNumpy.ipynb
    UsingNumpy.rst
    [34mUsingNumpy_files[m[m/
    UsingPandas.ipynb
    UsingPandas.rst
    [34mUsingPandas_files[m[m/
    WorkingWithStructuredData.ipynb
    WorkingWithStructuredData.rst
    WritingParallelCode.ipynb
    WritingParallelCode.rst
    conf.py
    index.rst
    screenshot.png


.. code:: python

    cp IntroductionToPython.ipynb IP2.ipynb


.. code:: python

    ls


.. parsed-literal::

    AlgorithmicComplexity.ipynb
    AlgorithmicComplexity.rst
    [34mAlgorithmicComplexity_files[m[m/
    Animation.ipynb
    Animation.rst
    [34mAnimation_files[m[m/
    BlackBoxOptimization.ipynb
    BlackBoxOptimization.rst
    [34mBlackBoxOptimization_files[m[m/
    CUDAPython.ipynb
    CUDAPython.rst
    [34mCUDAPython_files[m[m/
    CalibratingODEs.ipynb
    CalibratingODEs.rst
    [34mCalibratingODEs_files[m[m/
    ComputationalStatisticsMotivation.ipynb
    ComputationalStatisticsMotivation.rst
    [34mComputationalStatisticsMotivation_files[m[m/
    ComputerArithmetic.ipynb
    ComputerArithmetic.rst
    [34mComputerArithmetic_files[m[m/
    CrashCourseInC.ipynb
    CrashCourseInC.rst
    DataProcessing-Solutions.ipynb
    DataProcessing-Solutions.rst
    [34mDataProcessing-Solutions_files[m[m/
    DataProcessing.ipynb
    DataProcessing.rst
    [34mDataProcessing_files[m[m/
    DistributedComputing.ipynb
    DistributedComputing.rst
    EM_algorithm.ipynb
    EM_algorithm.rst
    [34mEM_algorithm_files[m[m/
    FromCToPython.ipynb
    FromCToPython.rst
    FromCompiledToPython.ipynb
    FromCompiledToPython.rst
    FromJuliaToPython.ipynb
    FromJuliaToPython.rst
    [34mFromJuliaToPython_files[m[m/
    FromPythonToC.ipynb
    FromPythonToC.rst
    Functions-Solutions.rst
    Functions.ipynb
    Functions.rst
    FunctionsSolutions.ipynb
    FunctionsSolutions.rst
    GPUSAndCUDAC.rst
    [34mGPUs and Cuda C_files[m[m/
    GPUsAndCUDAC.ipynb
    [34mGPUsAndCUDAC_files[m[m/
    IP2.ipynb
    IP2.rst
    IPythonNotebookIntroduction.ipynb
    IPythonNotebookIntroduction.rst
    [34mIPythonNotebookIntroduction_files[m[m/
    IPythonNotebookPolyglot.ipynb
    IPythonNotebookPolyglot.rst
    [34mIPythonNotebookPolyglot_files[m[m/
    IntroductionToPython-Solutions.ipynb
    IntroductionToPython-Solutions.rst
    IntroductionToPython.ipynb
    IntroductionToPython.rst
    LinearAlgebraMatrixDecomp-WithSoutions.ipynb
    LinearAlgebraMatrixDecomp-WithSoutions.rst
    LinearAlgebraMatrixDecomp.ipynb
    LinearAlgebraMatrixDecomp.rst
    LinearAlgebraReview.ipynb
    LinearAlgebraReview.rst
    [34mLinearAlgebraReview_files[m[m/
    MCMC.ipynb
    MCMC.rst
    [34mMCMC_files[m[m/
    Makefile
    Makefile~
    MakingCodeFast.ipynb
    MakingCodeFast.rst
    [34mMakingCodeFast_files[m[m/
    MapReduce.ipynb
    MapReduce.rst
    MonteCarlo.ipynb
    MonteCarlo.rst
    [34mMonteCarlo_files[m[m/
    MultivariateOptimizationAlgortihms.ipynb
    MultivariateOptimizationAlgortihms.rst
    OptimizationInOneDimension.ipynb
    OptimizationInOneDimension.rst
    [34mOptimizationInOneDimension_files[m[m/
    Optimization_Bakeoff.ipynb
    Optimization_Bakeoff.rst
    PCA-Solutions.ipynb
    PCA-Solutions.rst
    [34mPCA-Solutions_files[m[m/
    PCA.ipynb
    PCA.rst
    [34mPCA_files[m[m/
    PyMC2.ipynb
    PyMC2.rst
    [34mPyMC2_files[m[m/
    PyMC3.ipynb
    PyMC3.rst
    [34mPyMC3_files[m[m/
    PyStan.ipynb
    PyStan.rst
    [34mPyStan_files[m[m/
    ResamplingAndMonteCarloSimulations.ipynb
    ResamplingAndMonteCarloSimulations.rst
    [34mResamplingAndMonteCarloSimulations_files[m[m/
    Spark.ipynb
    Spark.rst
    TextProcessing-Solutions.ipynb
    TextProcessing-Solutions.rst
    TextProcessing.ipynb
    TextProcessing.rst
    TextProcessingExtras.ipynb
    TextProcessingExtras.rst
    UsingNumpy-Solutions.ipynb
    UsingNumpy-Solutions.rst
    [34mUsingNumpy-Solutions_files[m[m/
    UsingNumpy.ipynb
    UsingNumpy.rst
    [34mUsingNumpy_files[m[m/
    UsingPandas.ipynb
    UsingPandas.rst
    [34mUsingPandas_files[m[m/
    WorkingWithStructuredData.ipynb
    WorkingWithStructuredData.rst
    WritingParallelCode.ipynb
    WritingParallelCode.rst
    conf.py
    index.rst
    screenshot.png


.. code:: python

    rm IP2.ipynb

.. code:: python

    ls



.. parsed-literal::

    AlgorithmicComplexity.ipynb
    AlgorithmicComplexity.rst
    [34mAlgorithmicComplexity_files[m[m/
    Animation.ipynb
    Animation.rst
    [34mAnimation_files[m[m/
    BlackBoxOptimization.ipynb
    BlackBoxOptimization.rst
    [34mBlackBoxOptimization_files[m[m/
    CUDAPython.ipynb
    CUDAPython.rst
    [34mCUDAPython_files[m[m/
    CalibratingODEs.ipynb
    CalibratingODEs.rst
    [34mCalibratingODEs_files[m[m/
    ComputationalStatisticsMotivation.ipynb
    ComputationalStatisticsMotivation.rst
    [34mComputationalStatisticsMotivation_files[m[m/
    ComputerArithmetic.ipynb
    ComputerArithmetic.rst
    [34mComputerArithmetic_files[m[m/
    CrashCourseInC.ipynb
    CrashCourseInC.rst
    DataProcessing-Solutions.ipynb
    DataProcessing-Solutions.rst
    [34mDataProcessing-Solutions_files[m[m/
    DataProcessing.ipynb
    DataProcessing.rst
    [34mDataProcessing_files[m[m/
    DistributedComputing.ipynb
    DistributedComputing.rst
    EM_algorithm.ipynb
    EM_algorithm.rst
    [34mEM_algorithm_files[m[m/
    FromCToPython.ipynb
    FromCToPython.rst
    FromCompiledToPython.ipynb
    FromCompiledToPython.rst
    FromJuliaToPython.ipynb
    FromJuliaToPython.rst
    [34mFromJuliaToPython_files[m[m/
    FromPythonToC.ipynb
    FromPythonToC.rst
    Functions-Solutions.rst
    Functions.ipynb
    Functions.rst
    FunctionsSolutions.ipynb
    FunctionsSolutions.rst
    GPUSAndCUDAC.rst
    [34mGPUs and Cuda C_files[m[m/
    GPUsAndCUDAC.ipynb
    [34mGPUsAndCUDAC_files[m[m/
    IP2.rst
    IPythonNotebookIntroduction.ipynb
    IPythonNotebookIntroduction.rst
    [34mIPythonNotebookIntroduction_files[m[m/
    IPythonNotebookPolyglot.ipynb
    IPythonNotebookPolyglot.rst
    [34mIPythonNotebookPolyglot_files[m[m/
    IntroductionToPython-Solutions.ipynb
    IntroductionToPython-Solutions.rst
    IntroductionToPython.ipynb
    IntroductionToPython.rst
    LinearAlgebraMatrixDecomp-WithSoutions.ipynb
    LinearAlgebraMatrixDecomp-WithSoutions.rst
    LinearAlgebraMatrixDecomp.ipynb
    LinearAlgebraMatrixDecomp.rst
    LinearAlgebraReview.ipynb
    LinearAlgebraReview.rst
    [34mLinearAlgebraReview_files[m[m/
    MCMC.ipynb
    MCMC.rst
    [34mMCMC_files[m[m/
    Makefile
    Makefile~
    MakingCodeFast.ipynb
    MakingCodeFast.rst
    [34mMakingCodeFast_files[m[m/
    MapReduce.ipynb
    MapReduce.rst
    MonteCarlo.ipynb
    MonteCarlo.rst
    [34mMonteCarlo_files[m[m/
    MultivariateOptimizationAlgortihms.ipynb
    MultivariateOptimizationAlgortihms.rst
    OptimizationInOneDimension.ipynb
    OptimizationInOneDimension.rst
    [34mOptimizationInOneDimension_files[m[m/
    Optimization_Bakeoff.ipynb
    Optimization_Bakeoff.rst
    PCA-Solutions.ipynb
    PCA-Solutions.rst
    [34mPCA-Solutions_files[m[m/
    PCA.ipynb
    PCA.rst
    [34mPCA_files[m[m/
    PyMC2.ipynb
    PyMC2.rst
    [34mPyMC2_files[m[m/
    PyMC3.ipynb
    PyMC3.rst
    [34mPyMC3_files[m[m/
    PyStan.ipynb
    PyStan.rst
    [34mPyStan_files[m[m/
    ResamplingAndMonteCarloSimulations.ipynb
    ResamplingAndMonteCarloSimulations.rst
    [34mResamplingAndMonteCarloSimulations_files[m[m/
    Spark.ipynb
    Spark.rst
    TextProcessing-Solutions.ipynb
    TextProcessing-Solutions.rst
    TextProcessing.ipynb
    TextProcessing.rst
    TextProcessingExtras.ipynb
    TextProcessingExtras.rst
    UsingNumpy-Solutions.ipynb
    UsingNumpy-Solutions.rst
    [34mUsingNumpy-Solutions_files[m[m/
    UsingNumpy.ipynb
    UsingNumpy.rst
    [34mUsingNumpy_files[m[m/
    UsingPandas.ipynb
    UsingPandas.rst
    [34mUsingPandas_files[m[m/
    WorkingWithStructuredData.ipynb
    WorkingWithStructuredData.rst
    WritingParallelCode.ipynb
    WritingParallelCode.rst
    conf.py
    index.rst
    screenshot.png


We can make all the above system calls in one cell, by using the cell
magic, %%system

.. code:: python

    %%system
    cp IntroductionToPython.ipynb  IP2.ipynb
    ls
    rm IP2.ipynb
    ls




.. parsed-literal::

    ['AlgorithmicComplexity.ipynb',
     'AlgorithmicComplexity.rst',
     'AlgorithmicComplexity_files',
     'Animation.ipynb',
     'Animation.rst',
     'Animation_files',
     'BlackBoxOptimization.ipynb',
     'BlackBoxOptimization.rst',
     'BlackBoxOptimization_files',
     'CUDAPython.ipynb',
     'CUDAPython.rst',
     'CUDAPython_files',
     'CalibratingODEs.ipynb',
     'CalibratingODEs.rst',
     'CalibratingODEs_files',
     'ComputationalStatisticsMotivation.ipynb',
     'ComputationalStatisticsMotivation.rst',
     'ComputationalStatisticsMotivation_files',
     'ComputerArithmetic.ipynb',
     'ComputerArithmetic.rst',
     'ComputerArithmetic_files',
     'CrashCourseInC.ipynb',
     'CrashCourseInC.rst',
     'DataProcessing-Solutions.ipynb',
     'DataProcessing-Solutions.rst',
     'DataProcessing-Solutions_files',
     'DataProcessing.ipynb',
     'DataProcessing.rst',
     'DataProcessing_files',
     'DistributedComputing.ipynb',
     'DistributedComputing.rst',
     'EM_algorithm.ipynb',
     'EM_algorithm.rst',
     'EM_algorithm_files',
     'FromCToPython.ipynb',
     'FromCToPython.rst',
     'FromCompiledToPython.ipynb',
     'FromCompiledToPython.rst',
     'FromJuliaToPython.ipynb',
     'FromJuliaToPython.rst',
     'FromJuliaToPython_files',
     'FromPythonToC.ipynb',
     'FromPythonToC.rst',
     'Functions-Solutions.rst',
     'Functions.ipynb',
     'Functions.rst',
     'FunctionsSolutions.ipynb',
     'FunctionsSolutions.rst',
     'GPUSAndCUDAC.rst',
     'GPUs and Cuda C_files',
     'GPUsAndCUDAC.ipynb',
     'GPUsAndCUDAC_files',
     'IP2.ipynb',
     'IP2.rst',
     'IPythonNotebookIntroduction.ipynb',
     'IPythonNotebookIntroduction.rst',
     'IPythonNotebookIntroduction_files',
     'IPythonNotebookPolyglot.ipynb',
     'IPythonNotebookPolyglot.rst',
     'IPythonNotebookPolyglot_files',
     'IntroductionToPython-Solutions.ipynb',
     'IntroductionToPython-Solutions.rst',
     'IntroductionToPython.ipynb',
     'IntroductionToPython.rst',
     'LinearAlgebraMatrixDecomp-WithSoutions.ipynb',
     'LinearAlgebraMatrixDecomp-WithSoutions.rst',
     'LinearAlgebraMatrixDecomp.ipynb',
     'LinearAlgebraMatrixDecomp.rst',
     'LinearAlgebraReview.ipynb',
     'LinearAlgebraReview.rst',
     'LinearAlgebraReview_files',
     'MCMC.ipynb',
     'MCMC.rst',
     'MCMC_files',
     'Makefile',
     'Makefile~',
     'MakingCodeFast.ipynb',
     'MakingCodeFast.rst',
     'MakingCodeFast_files',
     'MapReduce.ipynb',
     'MapReduce.rst',
     'MonteCarlo.ipynb',
     'MonteCarlo.rst',
     'MonteCarlo_files',
     'MultivariateOptimizationAlgortihms.ipynb',
     'MultivariateOptimizationAlgortihms.rst',
     'OptimizationInOneDimension.ipynb',
     'OptimizationInOneDimension.rst',
     'OptimizationInOneDimension_files',
     'Optimization_Bakeoff.ipynb',
     'Optimization_Bakeoff.rst',
     'PCA-Solutions.ipynb',
     'PCA-Solutions.rst',
     'PCA-Solutions_files',
     'PCA.ipynb',
     'PCA.rst',
     'PCA_files',
     'PyMC2.ipynb',
     'PyMC2.rst',
     'PyMC2_files',
     'PyMC3.ipynb',
     'PyMC3.rst',
     'PyMC3_files',
     'PyStan.ipynb',
     'PyStan.rst',
     'PyStan_files',
     'ResamplingAndMonteCarloSimulations.ipynb',
     'ResamplingAndMonteCarloSimulations.rst',
     'ResamplingAndMonteCarloSimulations_files',
     'Spark.ipynb',
     'Spark.rst',
     'TextProcessing-Solutions.ipynb',
     'TextProcessing-Solutions.rst',
     'TextProcessing.ipynb',
     'TextProcessing.rst',
     'TextProcessingExtras.ipynb',
     'TextProcessingExtras.rst',
     'UsingNumpy-Solutions.ipynb',
     'UsingNumpy-Solutions.rst',
     'UsingNumpy-Solutions_files',
     'UsingNumpy.ipynb',
     'UsingNumpy.rst',
     'UsingNumpy_files',
     'UsingPandas.ipynb',
     'UsingPandas.rst',
     'UsingPandas_files',
     'WorkingWithStructuredData.ipynb',
     'WorkingWithStructuredData.rst',
     'WritingParallelCode.ipynb',
     'WritingParallelCode.rst',
     'conf.py',
     'index.rst',
     'screenshot.png',
     'AlgorithmicComplexity.ipynb',
     'AlgorithmicComplexity.rst',
     'AlgorithmicComplexity_files',
     'Animation.ipynb',
     'Animation.rst',
     'Animation_files',
     'BlackBoxOptimization.ipynb',
     'BlackBoxOptimization.rst',
     'BlackBoxOptimization_files',
     'CUDAPython.ipynb',
     'CUDAPython.rst',
     'CUDAPython_files',
     'CalibratingODEs.ipynb',
     'CalibratingODEs.rst',
     'CalibratingODEs_files',
     'ComputationalStatisticsMotivation.ipynb',
     'ComputationalStatisticsMotivation.rst',
     'ComputationalStatisticsMotivation_files',
     'ComputerArithmetic.ipynb',
     'ComputerArithmetic.rst',
     'ComputerArithmetic_files',
     'CrashCourseInC.ipynb',
     'CrashCourseInC.rst',
     'DataProcessing-Solutions.ipynb',
     'DataProcessing-Solutions.rst',
     'DataProcessing-Solutions_files',
     'DataProcessing.ipynb',
     'DataProcessing.rst',
     'DataProcessing_files',
     'DistributedComputing.ipynb',
     'DistributedComputing.rst',
     'EM_algorithm.ipynb',
     'EM_algorithm.rst',
     'EM_algorithm_files',
     'FromCToPython.ipynb',
     'FromCToPython.rst',
     'FromCompiledToPython.ipynb',
     'FromCompiledToPython.rst',
     'FromJuliaToPython.ipynb',
     'FromJuliaToPython.rst',
     'FromJuliaToPython_files',
     'FromPythonToC.ipynb',
     'FromPythonToC.rst',
     'Functions-Solutions.rst',
     'Functions.ipynb',
     'Functions.rst',
     'FunctionsSolutions.ipynb',
     'FunctionsSolutions.rst',
     'GPUSAndCUDAC.rst',
     'GPUs and Cuda C_files',
     'GPUsAndCUDAC.ipynb',
     'GPUsAndCUDAC_files',
     'IP2.rst',
     'IPythonNotebookIntroduction.ipynb',
     'IPythonNotebookIntroduction.rst',
     'IPythonNotebookIntroduction_files',
     'IPythonNotebookPolyglot.ipynb',
     'IPythonNotebookPolyglot.rst',
     'IPythonNotebookPolyglot_files',
     'IntroductionToPython-Solutions.ipynb',
     'IntroductionToPython-Solutions.rst',
     'IntroductionToPython.ipynb',
     'IntroductionToPython.rst',
     'LinearAlgebraMatrixDecomp-WithSoutions.ipynb',
     'LinearAlgebraMatrixDecomp-WithSoutions.rst',
     'LinearAlgebraMatrixDecomp.ipynb',
     'LinearAlgebraMatrixDecomp.rst',
     'LinearAlgebraReview.ipynb',
     'LinearAlgebraReview.rst',
     'LinearAlgebraReview_files',
     'MCMC.ipynb',
     'MCMC.rst',
     'MCMC_files',
     'Makefile',
     'Makefile~',
     'MakingCodeFast.ipynb',
     'MakingCodeFast.rst',
     'MakingCodeFast_files',
     'MapReduce.ipynb',
     'MapReduce.rst',
     'MonteCarlo.ipynb',
     'MonteCarlo.rst',
     'MonteCarlo_files',
     'MultivariateOptimizationAlgortihms.ipynb',
     'MultivariateOptimizationAlgortihms.rst',
     'OptimizationInOneDimension.ipynb',
     'OptimizationInOneDimension.rst',
     'OptimizationInOneDimension_files',
     'Optimization_Bakeoff.ipynb',
     'Optimization_Bakeoff.rst',
     'PCA-Solutions.ipynb',
     'PCA-Solutions.rst',
     'PCA-Solutions_files',
     'PCA.ipynb',
     'PCA.rst',
     'PCA_files',
     'PyMC2.ipynb',
     'PyMC2.rst',
     'PyMC2_files',
     'PyMC3.ipynb',
     'PyMC3.rst',
     'PyMC3_files',
     'PyStan.ipynb',
     'PyStan.rst',
     'PyStan_files',
     'ResamplingAndMonteCarloSimulations.ipynb',
     'ResamplingAndMonteCarloSimulations.rst',
     'ResamplingAndMonteCarloSimulations_files',
     'Spark.ipynb',
     'Spark.rst',
     'TextProcessing-Solutions.ipynb',
     'TextProcessing-Solutions.rst',
     'TextProcessing.ipynb',
     'TextProcessing.rst',
     'TextProcessingExtras.ipynb',
     'TextProcessingExtras.rst',
     'UsingNumpy-Solutions.ipynb',
     'UsingNumpy-Solutions.rst',
     'UsingNumpy-Solutions_files',
     'UsingNumpy.ipynb',
     'UsingNumpy.rst',
     'UsingNumpy_files',
     'UsingPandas.ipynb',
     'UsingPandas.rst',
     'UsingPandas_files',
     'WorkingWithStructuredData.ipynb',
     'WorkingWithStructuredData.rst',
     'WritingParallelCode.ipynb',
     'WritingParallelCode.rst',
     'conf.py',
     'index.rst',
     'screenshot.png']



But magics are much more than system calls! We can even use R from
within the IPython notebook if you install the rpy2 package

.. code:: bash

    pip install rpy2

Python as Glue
--------------

.. code:: python

    %load_ext rpy2.ipython 

.. code:: python

    %matplotlib inline

.. code:: python

    %%R
    library(lattice) 
    attach(mtcars)
    
    # scatterplot matrix 
    splom(mtcars[c(1,3,4,5,6)], main="MTCARS Data")



.. image:: IPythonNotebookIntroduction_files/IPythonNotebookIntroduction_27_0.png


Matlab works too:

.. code:: bash

    pip install pymatbridge

.. code:: python

    !pip install --upgrade pymatbridge


.. parsed-literal::

    Requirement already up-to-date: pymatbridge in /Users/cliburn/anaconda/lib/python2.7/site-packages
    Cleaning up...


.. code:: python

    import pymatbridge as pymat
    ip = get_ipython()
    pymat.load_ipython_extension(ip)


.. parsed-literal::

    Starting MATLAB on ZMQ socket ipc:///tmp/pymatbridge
    Send 'exit' command to kill the server
    .MATLAB started and connected!


.. parsed-literal::

    /Users/cliburn/anaconda/lib/python2.7/site-packages/IPython/nbformat/current.py:19: UserWarning: IPython.nbformat.current is deprecated.
    
    - use IPython.nbformat for read/write/validate public API
    - use IPython.nbformat.vX directly to composing notebooks of a particular version
    
      """)


.. code:: python

    %%matlab
    
    xgv = -1.5:0.1:1.5;
    ygv = -3:0.1:3;
    [X,Y] = ndgrid(xgv,ygv);
    V = exp(-(X.^2 + Y.^2));
    surf(X,Y,V)
    title('Gridded Data Set', 'fontweight','b');



.. image:: IPythonNotebookIntroduction_files/IPythonNotebookIntroduction_31_0.png


.. code:: python

    ! pip install oct2py


.. parsed-literal::

    Requirement already satisfied (use --upgrade to upgrade): oct2py in /Users/cliburn/anaconda/lib/python2.7/site-packages
    Cleaning up...


.. code:: python

    %load_ext oct2py.ipython

.. code:: python

    %%octave
    
    A = reshape(1:4,2,2); 
    b = [36; 88];
    A\b
    [L,U,P] = lu(A)
    [Q,R] = qr(A)
    [V,D] = eig(A)



.. parsed-literal::

    ans =
    
           60
           -8
    
    L =
    
      1.00000  0.00000
      0.50000  1.00000
    
    U =
    
            2        4
            0        1
    
    P =
    
    Permutation Matrix
    
       0   1
       1   0
    
    Q =
    
      -0.44721  -0.89443
      -0.89443  0.44721
    
    R =
    
      -2.23607  -4.91935
      0.00000  -0.89443
    
    V =
    
      -0.90938  -0.56577
      0.41597  -0.82456
    
    D =
    
    Diagonal Matrix
    
      -0.37228        0
            0  5.37228


Python <-> R <-> Matlab <-> Octave
----------------------------------

.. code:: python

    import pandas as pd
    import numpy as np
    import statsmodels.api as sm 
    from pandas.tools.plotting import scatter_matrix

.. code:: python

    # First we will load the mtcars dataset and do a scatterplot matrix
    
    mtcars = sm.datasets.get_rdataset('mtcars')
    df = pd.DataFrame(mtcars.data)
    scatter_matrix(df[[0,2,3,4,5]], alpha=0.3, figsize=(8, 8), diagonal='kde', marker='o');



.. image:: IPythonNotebookIntroduction_files/IPythonNotebookIntroduction_37_0.png


.. code:: python

    # Next we will do the 3D mesh
    
    xgv = np.arange(-1.5, 1.5, 0.1)
    ygv = np.arange(-3, 3, 0.1)
    [X,Y] = np.meshgrid(xgv, ygv)
    V = np.exp(-(X**2 + Y**2))
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0.25)
    plt.title('Gridded Data Set');



.. image:: IPythonNotebookIntroduction_files/IPythonNotebookIntroduction_38_0.png


.. code:: python

    # And finally, the matrix manipulations
    
    import scipy
    
    A = np.reshape(np.arange(1, 5), (2,2))
    b = np.array([36, 88])
    ans = scipy.linalg.solve(A, b)
    P, L, U = scipy.linalg.lu(A)
    Q, R = scipy.linalg.qr(A)
    D, V = scipy.linalg.eig(A)
    print 'ans =\n', ans, '\n'
    print 'L =\n', L, '\n'
    print "U =\n", U, '\n'
    print "P = \nPermutation Matrix\n", P, '\n'
    print 'Q =\n', Q, '\n'
    print "R =\n", R, '\n'
    print 'V =\n', V, '\n'
    print "D =\nDiagonal matrix\n", np.diag(abs(D)), '\n'


.. parsed-literal::

    ans =
    [ 16.  10.] 
    
    L =
    [[ 1.          0.        ]
     [ 0.33333333  1.        ]] 
    
    U =
    [[ 3.          4.        ]
     [ 0.          0.66666667]] 
    
    P = 
    Permutation Matrix
    [[ 0.  1.]
     [ 1.  0.]] 
    
    Q =
    [[-0.31622777 -0.9486833 ]
     [-0.9486833   0.31622777]] 
    
    R =
    [[-3.16227766 -4.42718872]
     [ 0.         -0.63245553]] 
    
    V =
    [[-0.82456484 -0.41597356]
     [ 0.56576746 -0.90937671]] 
    
    D =
    Diagonal matrix
    [[ 0.37228132  0.        ]
     [ 0.          5.37228132]] 
    


More Glue: Julia and Perl
-------------------------

Using Julia
~~~~~~~~~~~

.. code:: python

    %load_ext julia.magic


.. parsed-literal::

    Initializing Julia interpreter. This may take some time...


.. code:: python

    %%julia
    1 + sin(3)




.. parsed-literal::

    1.1411200080598671



.. code:: python

    %%julia
    s = 0.0
    for n = 1:2:10000
        s += 1/n - 1/(n+1)
    end
    s # an expression on the last line (if it doesn't end with ";") is printed as "Out"




.. parsed-literal::

    0.6930971830599458



.. code:: python

    %%julia
    f(x) = x + 1
    f([1,1,2,3,5,8])




.. parsed-literal::

    [2, 2, 3, 4, 6, 9]



Using Perl
~~~~~~~~~~

.. code:: python

    %%perl
    
    use strict;
    use warnings;
     
    print "Hello World!\n";


.. parsed-literal::

    Hello World!


We hope these give you an idea of the power and flexibility this
notebook environment provides!
