
# MAIN100-10
### Euqal
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Equal/fedAVG-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Equal/fedProx-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Equal/fixed-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Equal/Upperbound.sh

### Featured
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Featured/fedAVG-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Featured/fedProx-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Featured/fixed-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Featured/Upperbound.sh

### Pareto
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Pareto/fedAVG-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Pareto/fedProx-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Pareto/fixed-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Pareto/Upperbound.sh

### Quantitative
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Quantitative/fedAVG-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Quantitative/fedProx-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Quantitative/fixed-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Quantitative/Upperbound.sh

### Unequal
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Unequal/fedAVG-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Unequal/fedProx-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Unequal/fixed-100.sh
qsub -g gad50699 ../experiment/FMNIST-100-10-01/Main/Unequal/Upperbound.sh

