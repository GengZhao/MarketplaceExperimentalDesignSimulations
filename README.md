# Marketplace Experimental Design Simulations

## Summary

This repository offers basic simulation programs and building blocks for simulating marketplace experiments (aka A/B test) in a finite static model.

The library implements `Marketplace` as a basic class, which configures a marketplace model (market size, booking rate, etc.) and executes a simple random booking process.

Other `.cc` files are simulation bundles built on the `Marketplace` class. Note that these simulations are multi-threaded and may consume large CPU time and/or memory.

## Usage
Each `<experiment>.cc` file can be executed as is after running `make <experiment>`. Parameters are set within the source file (something I should consider building a more friendly interface for), with the typical convention that `n` is the number of listings, `m` is the number of customers, `phi_0` and `phi_1` are consideration probabilities in control and treatment conditions, `a_c` and `a_l` are treatment proportions for customers and listings. There is usually a step size or number of steps parameter that dictates how many iterations to run (often with exponentially growing market size/market balance or linearly growing treatment ratios). Also, beware of the `nThrds` parameter which sets the number of threads to be used.

The raw output is printed into a file, which can be subsequently analyzed using Python scripts.
