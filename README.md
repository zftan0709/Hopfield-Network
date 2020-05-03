# **Hopfield Network Simulation**
---

## Introduction
Hopefield Network is a type of recurrent neural network and associative memory which is different from classic pattern.
Hopfield network can be used to store patterns and recover patterns from distorted input. 
For instance, Hopfield network can recover image patterns from fuzzy input based on the patterns which is memorized beforehand. 
In Hopfield network, the symmetric weights ensure that the energy function decreases monotonically following the activation rule.

## Simulation
In this project, a Hopfield network model is built to reconstruct noisy image. To compare asynchronous method
and synchronous method, both update methods are performed in the program. This program
requires two parameter input to run the simulation which are the training image file directories
and number of training iteration. The program will then visualize the updated states from both
asynchronous update and synchronous update and then print their corresponding energy.
Finally, the energies from each iteration are plotted in a graph and is shown along with the
stable states for both methods.

### Example Input Images
In this code, the model is set to fixed amount of neurons. Therefore, can only accept 32x32 images
as shown below as input.  
<img src="/1.png" alt="Training1" title="Training Image 1" width="100" height="100" border="10"/>
<img src="/2.png" alt="Training2" title="Training Image 2" width="100" height="100" border="10"/>
<img src="/3.png" alt="Training3" title="Training Image 3" width="100" height="100" border="10"/> 
  
![Training4](/training&input.png)


## Instruction
**Numpy** and **Matplotlib** libraries are required to run the code. 
To execute the code, run the following command in terminal.
```
python hopfield.py -t IMAGE_DIRECTORIES -i NUMBER_OF_ITERATION
```
e.g.
```
python hopfield.py -t 1.PNG 2.PNG 3.PNG 4.PNG 5.PNG 6.PNG -i 100
```

## Result
Running the command from above will output a final result which includes the training curve for both asynchronous and synchronous method.
Additionally, the model weights will also be visualize in a figure as well. Example of output result is shown below.  
![Result](/result.png)

## Reference

[1] Schalkoff, Robert J. Artificial Neural Networks. McGraw-Hill, 1997.  
[2] Hopfield, John J. “Neural Networks and Physical Systems with Emergent Collective
Computational Abilities.” Feynman and Computation, pp. 7–19., doi:10.1201/9780429500459-2.