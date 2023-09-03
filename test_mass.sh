#!/bin/bash

DIR=sep1
SEED=100
for SAMPLE in LowMass HighMass; do
	echo $SAMPLE
	if [ "$SAMPLE" = HighMass ]; then SEED=1000; fi
	python train.py -C configs/dijetPair${SAMPLE}.py --outf dijetPair${SAMPLE}_${DIR}_1 --epochs 50 --sampling undersample --learning-rate 0.01 --random-seed $SEED
	python train.py -C configs/dijetPair${SAMPLE}.py --outf dijetPair${SAMPLE}_${DIR}_2 --epochs 100 --sampling undersample --learning-rate 0.001 --random-seed $SEED --continue-train dijetPair${SAMPLE}_${DIR}_1/models
	python train.py -C configs/dijetPair${SAMPLE}.py --outf dijetPair${SAMPLE}_${DIR}_3 --epochs 50 --sampling undersample --learning-rate 0.0001 --random-seed $SEED --continue-train dijetPair${SAMPLE}_${DIR}_2/models
	python test.py -C configs/dijetPair${SAMPLE}.py --outf dijetPair${SAMPLE}_${DIR}_3 --name testGauss --calibrate --calib-gauss
	python analyze.py -C configs/dijetPairAll.py --outf dijetPair${SAMPLE}_${DIR}_3 --name plotsPair --theory "" --calib-source dijetPair${SAMPLE}_${DIR}_3/testGauss/calibrations.py --extras Mjj_msortedP1_high Mjj_msortedP1_low Mjj_msortedP2_high Mjj_msortedP2_low Mjj_msortedP3_high Mjj_msortedP3_low --pair
	python analyze.py -C configs/dijetPair.py --outf dijetPair${SAMPLE}_${DIR}_3 --name plotsPairAvg --theory dR_M Truth_M_avg Truth_high_M_avg Truth_avg_M_avg --xmin 250 --xmax 1250 --bins 50 --calib-source dijetPair${SAMPLE}_${DIR}_3/plotsPair/calibrations.py
done

