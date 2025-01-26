### BULK 2 (ID=539)
Original E_min / E_max Ratio ~ 1/40
Original Sim Poisson's Ratio ~ 0.0
Original Output Sim Poisson's Ratio ~ 0.571

Adjusted Input Poisson's Ratio ~ 0.1
Adjusted Output Metamaterial Poisson's Ratio ~ 0.607

DIC MATLAB Poisson's Ratio ~ 0.2
Approx. E_min / E_max Ratio to have sim match reality ~ 1/5
E Adjusted Poisson's Ratio ~ 0.193

E_min 1/10 Poisson's Ratio ~ 0.31

### BULK 1 (ID=510) (Appox. Isotropic Auxetic)
Original E_min / E_max Ratio ~ 1/60
Original Sim Poisson's Ratio ~ 0.0
Original Sim Poisson's Ratio ~ -0.303

Adjusted Input Poisson's Ratio ~ 0.1
Adjusted Output Metamaterial Poisson's Ratio ~ -0.28

DIC MATLAB Poisson's Ratio ~ 0.03
Approx. E_min / E_max Ratio to have sim match reality ~ 1/10
E Adjusted Poisson's Ratio ~ 0.05

### BULK 1 (ID=529) (Ortho. Auxetic)
Original E_min / E_max Ratio ~ 1/40
Original Sim Poisson's Ratio ~ 0.0
Original Sim Poisson's Ratio ~ -0.275

Adjusted Input Poisson's Ratio ~ 0.1
Adjusted Output Metamaterial Poisson's Ratio ~ -0.25

DIC MATLAB Poisson's Ratio ~ 0.03
Approx. E_min / E_max Ratio to have sim match reality ~ 1/10
E Adjusted Poisson's Ratio ~ 0.0225

### VERT 1 (ID=553) (Vertical ZPR)
Original E_min / E_max Ratio ~ 1/40
Original Input Poisson's Ratio ~ 0.0
Original Metamaterial Output Poisson's Ratio ~ 0.017

Adjusted Input Poisson's Ratio ~ 0.1
Adjusted Metamaterial Output Poisson's Ratio ~ 0.037

DIC MATLAB Poisson's Ratio ~ 0.06
Approx. E_min / E_max Ratio to have sim match reality ~ 1/30
E Adjusted Poisson's Ratio ~ 0.0577

We measured at 1/40
Simulation didn't match reality
Correct simulation with fudge to get closer
That number is E_min = 1/10

Then we rerun TO with 1/10 and map property space

** Check if we rerun with E_min=1/10 and nu=1/10 
TO with the same seeds as the picked 

(1) We printed homogeneous foams. Measure E ratio to be 1:40, nu=0.1
(2) We generate structures using this assumption
(3) These structures don't match simulation
(4) We correct for this by adjusting the E ratio for the sim to 1:10
(5) Then we check the predicted property space coverage