# DisCGS

This code performs the distributed inference for Multinomial Dirichlet Process Mixture Model inference.

### Requirements

* Scala 2.13.0
* Spark 3.3.0
  
See src/pom.xml file for Scala and Spark dependencies.

### Building

The script build.sh is provided to build an executable jar containing all the dependencies. 
Use the following command to build it: 
```
/bin/bash build.sh
```

### Running 

In order to run the built jar use the following code:

```
scala -J-Xmx1024m target/DisDPMM_2.13-1.0-jar-with-dependencies.jar <dataset name> <gamma> <number of iterations> <if distributed> <number of workers>
```

Example of execution:

```
scala -J-Xmx1024m target/DisDPMM_2.13-1.0-jar-with-dependencies.jar Tweet 0.02 10 true 2
```
The above code will run DisCGS for  10 iterations on Tweet dataset using 2 workers.

The datasets used in the paper are provided in dataset file.

The program will output the runtime (in seconds), ARI, NMI and the number of inferred clusters:

```
Results:
ARI               : 0.673
NMI               : 0.857
Inferred clusters : 107
Gamma             : 0.02
True clusters     : 89
Running time      : 3.641
```
