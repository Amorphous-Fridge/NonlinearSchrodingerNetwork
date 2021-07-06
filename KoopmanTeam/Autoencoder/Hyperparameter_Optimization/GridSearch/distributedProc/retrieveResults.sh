awk -F',' '{m=$1;for(i=1;i<=NF;i++)if($i<m)m=$i;print m}' test.txt #finds the lowest value in a comma-separated row
grep -E -o --color '(loss: [0-9.]{6})' testloss.txt
grep -E -o --color '(loss: [0-9.]{6})' testloss.txt | grep -E -o --color '[0-9.]{6}'
#get min loss in a specific file
grep -E -o --color '(loss: [0-9.]{6})' testloss.txt | grep -E -o --color '[0-9.]{6}' | awk 'BEGIN{a=10}{if ($1<0+a) a=$1} END{print a}'

grep -E -o --color '(loss: [0-9.]{6}e-0[456])' slurm-42412142_23.out | grep -E -o --color '([0-9.]{6}e-0[456])' | awk 'BEGIN{a=10}{if ($1<0+a) a=$1} END{print a}'

grep -E -o --color '(loss: [0-9.]{6}e-0[456])' slurm-42412142_64.out | grep -E -o --color '([0-9.]{6}e-0[456])' | awk 'BEGIN{a=10}{if ($1<0+a) a=$1} END{print NR, a}' >> results.txt


3.8311e-04 63





1-24
#slurm file codes
#42412137	1 layer
#42412142	2 layers
#42412173	3 layers
#42412206	4 layers
#42412282	5 layers
#42412319	6 layers
