# Marketplace Experimental Design Simulations

## Summary

This repository offers basic simulation programs and building blocks for simulating marketplace experiments (aka A/B test) in a finite static model.

The library implements `Marketplace` as a basic class, which configures a marketplace model (market size, booking rate, etc.) and executes a simple random booking process.

Other `.cc` files are simulation bundles built on the `Marketplace` class. Note that these simulations are multi-threaded and may consume large CPU time and/or memory.
