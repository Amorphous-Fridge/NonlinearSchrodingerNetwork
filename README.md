# NonlinearSchrodingerNetwork

## TODO

- Test nonlinearity of evolution function
    - $F(\phi(v) + \phi(w)) \neq F(\phi(v)+\phi(w))$
        - Want the difference of the two sides to approach something non-zero
    - Test on points for which $\phi(v),\phi(w)$ are in the evolution function's training set and $\phi(v)+\phi(w)$ is within some epsilon of another point in our (compressed) training set.
    - Evaluate nonlinearity as a function of training data and as a function of the timestep

- Explore the dataspace of the nonlinear function
    - Evaluate nonlinearity and accuracy as a function of initial conditions and timestep

- (Side project) Determine what information must be conserved when compressing from 4 dimensions to 2
    - We want the $\sim$ such that $(\alpha+i\beta, \gamma+i\delta)\sim(\tilde{\alpha}+i\tilde{\beta},\tilde{\gamma}+i\tilde{\delta})$ on the bloch sphere, where the RHS is the autoencoded state given by the LHS.
    - Repeat essentially everything we've done for a 4-2-4 compression


