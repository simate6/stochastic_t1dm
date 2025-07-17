# Stochastic Virtual Population in Type 1 Diabetes

Accurate, reliable, and efficient estimation of blood glucose dynamics from real-world data is challenging due to the time-varying nature, high uncertainty, and nonlinear interplay of complex processes. This repository contains a stochastic representation of a virtual population by fitting a hierarchical Bayesian model. In total, 500 24h-long sequences, 50 from each of the 10 patients with type 1 diabetes on multiple daily injection therapy is used. Uncertainty is modeled on multiple levels, in physiology and in self-reported events, and intra- and interday variability, and the effect of physical activity are take into account as well. The root-mean-square error between the glucose measurements and the mean of the posterior predictive distribution using the fitted low-rank multivariate normal guide is 12.44 mg/dL. The posterior distributions can be used to simulate realistic intra-, and interday variability in terms of the investigated patient cohort.

## Implementation

[Model implementation and fitting](run_svi_t1dm_population.ipynb)

* The probabilistic model is implemented using Numpyro, the deterministic submodel using JAX and the model is end-to-end differentiable.
* The hierarchical model is fitted using an auto low-rank multivariate normal guide with stochastic variational inference.
* The total fitting process using 15 samples for the approximation of the evidence lower bound, with 80,000 iterations, takes 2h:25min on a Google TPU v6e-1.

[Plots](/figures)

* Shows the posterior predictive distributions for the 500, 24h-long sample.

## Dataset

10 participants (HbA1c of 7.1±0.6%, age of 35±7.1 years, and body weight of 73±9.4 kg) with type 1 diabetes wore a CGM device (Medtronic Guardian 3), activity tracker, and reported their daily insulin injections, meal intakes, and physical activities in a smartphone application over several weeks. All of the participants were on MDI therapy, and asked to follow their regular routine, thus the data collection was under free-living conditions. Meal intakes could be selected from a large database of the Diabtrend smartphone application with predefined macronutritional composition or they could manually entry them. Physical activities were self-reported in the application, heart rate was measured by Xiaomi Miband 6 activity tracker.
