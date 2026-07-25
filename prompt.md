2026.07.25

# Revsie code structure, unit the optimization method, rerun Figure (b)

Revise the codes. 1. Set the optimizations to be identical optimizer settings In 'plot_fixed_tilt.py' and 'plot_ascending_tilt.py', i.e. same hyperparameters (learning-rate decay and tilt penalty ). Be mindful to choose a set of good hyperparameters to make both results nice. 2. Revise the structure of all codes and functions in the project. I want the following structure: File 1, named 'max_cut.py', has all functions related to the maxcut problem. File 2, named 'qaoa.py', has all functions related to the QAOA circuit Ansatz. File 3, named 'qtl.py', has all functions related to a single step execution of the entire procedure with input of problem specifications, parameters, settings, and output of those data and results in the corresponding 'csv' files. File 4, named 'fixed_tilt.py', corresponds to a simple plot function with the csv data as input and the Figure (a) as output. File 5, named 'ascending_tilt.py', corresponds to a simple plot function with the csv data as input and the Figure (b) as output. File 6, named 'scale_benchmark.py' with the csv data as input and the Figure (c) as output. If more files are necessaary, it's also okay to have more files to make the project sturcture clear and logical.

Now run the program to regenerate figure (a), (b), and (c).

Set the plotting styles of new figures to be the same as those old ones, including linestyles, etc. And regenerate the figures.

For Figure (a), please keep the original 'matching' function to fit the curves and plot those curves.

For Figure (b), also keep the original 'matching' function to fit the curves and plot those curves.

Since we have specified the optimization settings to be identical, we should rerun the entire program to generate data and then plot the figures.

Revise the codes. In the Figure (b), the matching function is no longer suitable. Change it to another function.

Revise the codes. Change the parameters including learning rate, decay power, decay offset, momentum, tilt penalty, and gradient clipping to be those parameters that have the best performance for the ascending gamma results. Can refer to those old parameters.


# Add comparison of parameter-shift rule and finite-difference method 

Revise the codes. Generate a new 'parameter_shift_comparison.py' file. It contains a particular maxcut problem solving using QAOA algorithm with quantum tilted loss function. Then add another function usng parameter-shift rule to do the parameter updates. Record the problem specifications in a 'parameter_shift_comparison.csv' file. Then solve the problem using QAOA algorithm with quantum tilted loss function with the finite difference method to do the parameter updates. Record the losses for every iterations of the parameter updates in the same csv file. Then solve the problem using QAOA algorithm with quantum tilted loss function with the new parameter-shift rule to do the parameter updates. Record the losses for every iterations of the parameter updates in the same csv file. Then Plot a figure with x axis as per iteration and y axis as the mean final ratio (loss function). Plot a line showing the loss functions with the iterations using the finite difference optimimizer. Plot another line showing the loss functions with the iterations using the new parameter-shift rule optimization method. Make the figure more academic style.

Here, we use finite shots and a bit large finite difference step in order to visualize the difference of these two methods. Also plot a figure showing the errors of the current parameter vector versus the optimal parameter vecter for both methods.
Updated and reran the comparison with:
- 2,000 shots per circuit evaluation
- Finite-difference step 0.05
- 60 optimization iterations
- Analytic L-BFGS-B reference-optimal parameter vector

The figure of mean final ratio is overlapped with the subfigure. Please make the subfigure more down and left. Make the figure more academic style.


# Add CVaR for comparison, fixed alpha

Revise the codes. Generate a new 'cvar_comparison.py' file. It contains maxcut problem solving using QAOA algorithm with the CVaR as loss function. This is similar to the process of running the algorithms and generate figure (b). Now the shots are set to be 5000. Run numerical simulations to solve problems using several fixed alpha in CVaR. Run numerical simulations to solve problems using several ascending alpha in CVaR. Record the results in a new 'CVaR.csv' file. Plot a figure (d) named 'CVAR' in both JPG and PDF format that is similar to Figure (b) but with the loss function changed to CVaR and the x axis is changed to be alphas. Then use a matching function to fit the results in relation to alpha. Compare this function and the matching function to fit the quantum tilted loss results in Figure (b). And plot these two functions together in another figure named 'performance_matching_function' in both JPG and PDF format. Make the figure more academic style.










