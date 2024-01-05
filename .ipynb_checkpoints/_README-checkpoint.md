---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.3
  nbformat: 4
  nbformat_minor: 5
---

::: {#af65c08c-6403-4492-ac1a-cbf921bf3896 .cell .markdown}
# 1. Software Environment Setups {#1-software-environment-setups}

## Lauch Jupyter Notebook in \"pyenv version 2.3.27\" and \"python version 3.10.13\" {#lauch-jupyter-notebook-in-pyenv-version-2327-and-python-version-31013}

-   Caution: \"Do not lauch Jupyeter Notebook in Anaconda because there
    is a kernel break down problem in Anaconda.\"

-   Python packages are listed in \"Part 2: Code\" of the ACC form.

-   Terminal Console

    -   install pyenv
    -   pyen activate \[your python environment\]
    -   jupyter notebook

-   Start your jupyter notebook in this foler directory
:::

::: {#68f85cb3 .cell .markdown}
# 2. Running Environment Setups {#2-running-environment-setups}

*IMPORTANT NOTICE* Please ensure that \"\[PATH TO
HETERORL\]/HeteroRL/hetero\" is included in your Python search PATH.
This step is crucial to enable Python to locate and access the functions
defined in the files within \"\[PATH TO HETERORL\]/HeteroRL/hetero.\"
The specific method for achieving this may vary depending on your
computer\'s individual system settings. This is a common Python issue
for which solutions can be readily found on the internet.

-   Start your jupyter notebook in this foler directory

OR

-   Run the code in the cell below to set your working directory as this
    folder \"HeteroRL\"
-   So that you can access the \"hetero\" folder as a library
:::

::: {#64cfe273 .cell .code execution_count="1"}
``` python
WORK_DIR = "[PATH TO HETERORL]/HeteroRL"

import os
os.chdir(WORK_DIR)
```
:::

::: {#02fc1d18 .cell .markdown}
# 3. Reproducing Results in Section 5. Simulations {#3-reproducing-results-in-section-5-simulations}
:::

::: {#723a4106 .cell .markdown}
# 3.1 Result I: Coefficients of the value function of a given policy $\pi$ {#31-result-i-coefficients-of-the-value-function-of-a-given-policy-pi}

This part of results is related to policy evaluation \-- coefficient
estimation.

-   Page 26 of the paper

-   Figure 2 is produced by running
    \`\`QL-Hetero-20230602-LINEAR-SPLIT.ipynb\'\'. The result is in the
    last cell of the ipynb.

    -   Restart the kernel and run all in the ipynb.
:::

::: {#d555ca89 .cell .markdown}
# 3.2 Result II: Confidence Intervels of the integrated value of a given policy $\pi$ {#32-result-ii-confidence-intervels-of-the-integrated-value-of-a-given-policy-pi}

This part of results is related to policy evaluation \-- the confidence
interval of coefficient estimation.

-   Reproduce Figure 3 and Table 1 through the following steps:

    -   Restart kernel and run all for each of the following notebooks
        in order. The name of the ipynb suggests the setting of the
        experiments.

        -   QL-Hetero-20230602-LINEAR-N20-T20.ipynb \--\> \"Linear
            setting with N = 20 and T = 20\"
        -   QL-Hetero-20230602-LINEAR-N20-T30.ipynb \--\> \"Linear
            setting with N = 20 and T = 30\"
        -   QL-Hetero-20230602-LINEAR-N20-T40.ipynb \--\> \"Linear
            setting with N = 20 and T = 40\"
        -   QL-Hetero-20230602-LINEAR-N50-T20.ipynb \--\> \"Linear
            setting with N =50 and T = 20\"
        -   QL-Hetero-20230602-LINEAR-N50-T30.ipynb \--\> \"Linear
            setting with N = 50 and T = 30\"
        -   QL-Hetero-20230602-LINEAR-N50-T40.ipynb \--\> \"Linear
            setting with N = 50 and T = 40\"
        -   QL-Hetero-20230602-LINEAR-N100-T20.ipynb \--\> \"Linear
            setting with N = 100 and T = 20\"
        -   QL-Hetero-20230602-LINEAR-N100-T30.ipynb \--\> \"Linear
            setting with N = 100 and T = 30\"
        -   QL-Hetero-20230602-LINEAR-N100-T40.ipynb \--\> \"Linear
            setting with N = 100 and T = 40\"

    -   The last cell of each notebook will output:

        -   MSE: Mean Squared Error
        -   ACL: Average Confidence Length
        -   ECP: Empirical Coverage Probability

        for two algorithms (ACPE and MVPE) under the setting of the
        notebook. For example, the last cell of
        \`\`QL-Hetero-20230602-LINEAR-N20-T30.ipynb\'\' outputs:

        -   ACPE results: Group1, Group 2 \<\-\-\-- ACPE results for
            each group 1 and Group 2, respectively.
        -   MSE: \[0.01998241 0.02120073\]
        -   ACL: \[0.64936136 0.64944401\]
        -   ECP: \[0.97 0.99\]
        -   ===
        -   MVPE results: \<\-\-\-- MVPE results for each group 1 and
            Group 2, respectively.
        -   MSE: \[12.82583 13.151297\]
        -   ACL: \[0.99122333\]
        -   ECP: \[0. 0.\]

    -   Compile all the numbers from all the ipynb notebook to produce:

        -   Figure 3 by using ECPs and \`\`plot_ecp.ipynb\'\' for
            plotting.
        -   Table 1 by typing all the numbers in the latex table.
:::

::: {#286d1f96 .cell .markdown}
# 3.3 Result III: Parametric optimal policy {#33-result-iii-parametric-optimal-policy}

This part of results are related to policy optimization.

-   Table 2 and Table 3 are obtained from
    \`\`QL-Hetero-20230602-LEGENDRE-PO.ipynb\'\'

-   Results in Table 3 are read off from output in

    -   Result of running Cell 8:
        \`\`MEAN_Y\_FINAL=1.0588678121566772\'\'
    -   Result of running Cell 9:
        \`\`MEAN_Y\_FINAL=0.9919090270996094\'\'
    -   Results of running Cell 10:
        -   Combined pol eval on cluser 0 = 0.3508417
        -   Combined pol eval on cluser 1 = -0.30345097

-   Results in Table 2 are read off from output in

    -   The section named \`\`Examing Specific Examples \--\> Results
        for Table 2\'\'
    -   Results of running the last two cells of the ipynb.
:::

::: {#7618d5f7 .cell .markdown}
# 4. Reproducing Results in Section 6. Real Data {#4-reproducing-results-in-section-6-real-data}

-   Run REALDATA_ALL_6.ipynb
:::

::: {#85d9ea91 .cell .markdown}
# 5. Reproducing Results in Appendix: Section A.3: Number of basis functions {#5-reproducing-results-in-appendix-section-a3-number-of-basis-functions}

-   Restart kernel and run all cells in
    \`\`LEGENDRE_DIFF_ORDER_V0_Fixed_Coeff_COMBINED.ipynb\'\'
-   Figure 6 are obtained in the respective Sections \"Plot Figure 6
    (Right)\" and \"Plot Figure 6 (Left)\"
:::

::: {#6f8ce638 .cell .markdown}
# 6. Reproducing Results in Appendix B: Additional Simulation Results {#6-reproducing-results-in-appendix-b-additional-simulation-results}

## 6.1 Section B.1: Non-linear reward, non-linear values, and non-Gaussian noise {#61-section-b1-non-linear-reward-non-linear-values-and-non-gaussian-noise}

-   Figure 7 in Appendix B, Section B.1 in the supplemental material is
    produced by running \`\`QL-Hetero-20230602-LEGENDRE-SPLIT.ipynb\'\'.
    The result is in the last cell of the ipynb.

    -   Restart the kernel and run all in the ipynb.

-   Figure 8 and Table 5 in Appendix B in the supplemental materials are
    produced by following the same procedure as described above with
    each of the following ipynb.

    -   Restart kernel and run all for each of the following notebooks
        in order. The name of the ipynb suggests the setting of the
        experiments.

        -   QL-Hetero-20230602-LEGENDRE-N20-T20.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N20-T30.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N20-T40.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N50-T20.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N50-T30.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N50-T40.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N100-T20.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N100-T30.ipynb
        -   QL-Hetero-20230602-LEGENDRE-N100-T40.ipynb

    -   The last cell of each notebook will output:

        -   MSE: Mean Squared Error
        -   ACL: Average Confidence Length
        -   ECP: Empirical Coverage Probability

        for two algorithms (ACPE and MVPE) under the setting of the
        notebook. For example, the last cell of
        \`\`QL-Hetero-20230602-LEGENDRE-N100-T40.ipynb\'\' outputs:

        -   ACPE results: Group1, Group 2 \<\-\-\-- ACPE results for
            each group 1 and Group 2, respectively.
        -   MSE: \[0.00090265 0.00061632\]
        -   ACL: \[0.11530798 0.11504481\]
        -   ECP: \[0.94 1. \]
        -   ===
        -   MVPE results: \<\-\-\-- MVPE results for each group 1 and
            Group 2, respectively.
        -   MSE: \[1.53640846 1.57656331\]
        -   ACL: \[0.14371708\]
        -   ECP: \[0. 0.\]

    -   Compile all the numbers from all the ipynb notebook to produce:

        -   Figure 8 by using ECPs and \`\`plot_ecp.ipynb\'\' for
            plotting.
        -   Table 5 by typing all the numbers in the latex table.
:::

::: {#9ac3e164 .cell .markdown}
## 6.2 Section B.2: Results for Homogenerous MDP {#62-section-b2-results-for-homogenerous-mdp}

-   Figure 9 is produced by:
    \`\`QL-Hetero-20230602-HOMO-LINEAR.ipynb\'\'

-   Figure 10 is prodcued by:
    \`\`QL-Hetero-20230602-HOMO-LEGENDRE.ipynb\'\'
:::

::: {#47027b9c .cell .markdown}
# 7. Core Files and Functions in \"hetero\" package {#7-core-files-and-functions-in-hetero-package}

Most variables and functions are self-explanatory in their names,
indicating their intended purposes. Comments are provided for some
functions or variables that are not self-explanatory.

-   
-   \"utils.py\" contains the utility functions to facilitate
    computation.

The program is designed to be flexible and with minimal redundancy.

## 7.1 Task Flow of ACPE {#71-task-flow-of-acpe}

For auto-clustered policy evaluation, the directly relevant file is
\"tasks.py\" for PE. The entire task flow is illustrated in
\"QL-Hetero-20230602-LINEAR-N100-T40.ipynb\"

-   At the beginning, data\'s label are assigned to each trajectory.
-   Algorithm 1 can be initiated from random betas, assigned betas, or
    betas estimated from individual trajectory from configuration.
-   Function \"beta_estimate_from_e2e_learning()\" is the end to end
    learning of Algorithm 1: ACPE. It outputs the value of each betas,
    clustering results, and centroid of each cluster. It calls the
    following core functions that implements the core calculations as
    described in Appendix Section A.1.
    -   \"impl = MCPImpl(data.N(), algo_config)\" \-\--\> implement MCP
        penalization
    -   \"beta_opt = BetaOptimizer(data, algo_config, pi_eval, impl,
        init_beta)\" \-\--\> class
    -   \"betas = beta_opt.compute()\" \-\--\> Compute ADMM and ALM for
        solving the minimizer beta
    -   \"learned_labels = group(betas, grouping_config)\" \-\--\>
        clustering based on estimated betas
    -   \"aligned_labels = align_binary_labels(learned_labels, truth)\"
        \-\--\> count the number of corrected aligned labels
-   

## 7.2 Task Flow of ACPI {#72-task-flow-of-acpi}

For auto-clustered policy iteration, the directly relevant file is
\"tasks.py\" for PE and \"offline_rl.py\" for PI. The entire task flow
is illustrated in \"QL-Hetero-20230602-LINEAR-PO.ipynb\"

-   Run \"7.1 Task Flow of ACPE\"
-   Obtain clustering results and update data label as its estimated
    group
-   Call \"offline_rl_mod.ActorCriticLeaner()\" for each group

## 7.3 Other Core Functions {#73-other-core-functions}

-   Function \"beta_estimate_from_nongrouped()\" \-\--\> is the classic
    pooled estimation
-   Function \"beta_estimate_from_e2e_learning()\" \-\--\> is the end to
    end learning of our ACPE (Policy Evaluation).
-   Class \"ActorCriticLeaner()\" \-\--\> implement ActorCritc
    Architecture
-   Class \"ActorLearner()\" \-\--\> implement Policy Iteration
:::

::: {#7b270868 .cell .markdown}
# The End
:::
