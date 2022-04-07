# Marketplace Experimental Design Simulations

## Summary

This repository offers basic simulation programs and building blocks for simulating marketplace experiments (aka A/B test) in a finite static model.

The library implements `Marketplace` as a basic class, which configures a marketplace model (market size, booking rate, etc.) and executes a simple random booking process.

Other `.cc` files are simulation bundles built on the `Marketplace` class. Note that these simulations are multi-threaded and may consume large CPU time and/or memory.

## Usage
Each `<experiment>.cc` file can be executed as is after running `make <experiment>`. Parameters are set within the source file (something I should consider building a more friendly interface for), with the typical convention that `n` is the number of listings, `m` is the number of customers, `phi_0` and `phi_1` are consideration probabilities in control and treatment conditions, `a_c` and `a_l` are treatment proportions for customers and listings. There is usually a step size or number of steps parameter that dictates how many iterations to run (often with exponentially growing market size/market balance or linearly growing treatment ratios). Also, beware of the common `nThrds` parameter which sets the number of threads to be used.

The raw output is printed into a file, which can be subsequently analyzed using Python scripts.

### Some experiments
Run `make lrcr_vary_design` compiles an executable `lrcr_vary_design` that evaluates different allocation rules in LR/CR designs, with the marketplace fixed. Make sure to check and/or set the parameters in the `lrcr_vary_design.cc` file. The program generates an output file located in the `Outputs/` directory named `lrcr_vary_design-<timestamp>.txt`. This raw data file contains outcomes of all the iterations (market characteristics, LR/CR estimators, etc.), and can be analyzed using the Python script `lrcr_design_analysis.py`. Example:
```
make lrcr_vary_design
./lrcr_vary_design
# Update the data file used in the Python script
./lrcr_design_analysis.py
```

Run `make eval_estimators` compiles an executable `eval_estimators` that evaluates different estimators (LR, CR, TSR) in different market settings (with preset allocation ratios and treatment effect). Make sure to check and/or set the parameters in the `eval_estimators.cc` file. The program generates an output file located in the `Outputs/` directory named `eval_estimators-<timestamp>.txt`. This raw data file contains outcomes of all the iterations (market characteristics, LR/CR estimators, etc.), and can be analyzed using the Python script `evaluate_estimators.py`.Example:
```
make eval_estimators
./eval_estimators
# Update the data file used in the Python script
./evaluate_estimators.py
```

Run `make conv_tradeoff` compiles an executable `conv_tradeoff` that showcases how the (normalized) bias and SD evolve as the market size scales up. Output file is named `bv_tradeoff-<timestamp>.txt`, and can be analyzed using the Python script `bias_sd_convergence.py`.

Run `make vary_treat` compiles an executable `vary_treat` that evaluates different estimators (LR, CR, TSR) as treatment effect varies (while fixing everything else). Output file is named `treatment_vary-<timestamp>.txt`, and can be analyzed using the Python script `vary_treatment_effect.py`.

