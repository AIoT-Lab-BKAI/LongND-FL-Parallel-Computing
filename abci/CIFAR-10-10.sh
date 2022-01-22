# Benchmark
### 2022-01-22 7:20
qsub -g gad50699 ../experiment/benchmark/cifar/equal/Upperbound-01.sh
qsub -g gad50699 ../experiment/benchmark/cifar/featured/Upperbound-01.sh
qsub -g gad50699 ../experiment/benchmark/cifar/pareto/Upperbound-01.sh
qsub -g gad50699 ../experiment/benchmark/cifar/quantitative/Upperbound-01.sh
qsub -g gad50699 ../experiment/benchmark/cifar/unequal/Upperbound-01.sh

# FEDRL
### 2022-01-22 7:21
qsub -g gad50699 ../experiment/ddpq/cifar/equal/fixed-01.sh
qsub -g gad50699 ../experiment/ddpq/cifar/featured/fixed-01.sh
qsub -g gad50699 ../experiment/ddpq/cifar/pareto/fixed-01.sh
qsub -g gad50699 ../experiment/ddpq/cifar/quantitative/fixed-01.sh
qsub -g gad50699 ../experiment/ddpq/cifar/unequal/fixed-01.sh
