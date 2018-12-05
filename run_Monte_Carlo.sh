#!/usr/bin/env bash
batch_size=1
res=()
for i in {1..500};do
    echo $i
    res+=( $(./dR.py | grep "Min Test Loss" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?') )
done
printf "%s\n" "${res[@]}" > ./monto_res/dev12/monte_res_IoT_"$batch_size"_"$1".txt
