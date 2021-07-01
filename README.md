# NonlinearSchrodingerNetwork

## TODO

- Find/Train best autoencoder architecture from grid search

- Get nonlinear evolution function to work using ideal compression
    - Timestep too long -> model jumps all over
    - Timestep too short -> model learns to not do anything (?)
    - Test using training data from many different initial conditions (on the same Hamiltonian)
        - Keep some initial condition evolutions aside for validation

- Train nonlinear evolution function using autoencoder for compression/decompression (FINAL STEP)

- Figure out what the next quantum system to work out should be
