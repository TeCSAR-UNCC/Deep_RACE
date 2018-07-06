#!/usr/bin/env bash
batch_size=2
res=()
for i in {1..250};do
    echo $i
    res+=( $(./dR.py | grep "Min Test Loss" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?') )
done
printf "%s\n" "${res[@]}" > ./monto_res/monte_res_sim_exp_"$batch_size"_"$1".txt