#!/bin/bash

#indata: json file of event mentions
#outdata: converted mrp file

#indata=data/raw/ace_en/"$split".json
#outdata=data/ie_graph_mrp/ace_en/"$split".mrp

# $1: dataset: ace_p, ace_pp, ere_p, ere_pp
# $2: language: en, zh, es

# evt-ent

for split in train test dev; do

    indata=data/raw/"$1"/"$2"/"$split".json
    outdata=data/ie_graph_mrp/"$1"_evt_ent/"$2"/"$split".mrp


    python mtool/main.py --strings --ids --read evt-ent --write mrp "$indata" "$outdata"
done;


# evt

for split in train test dev; do

    indata=data/raw/"$1"/"$2"/"$split".json
    outdata=data/ie_graph_mrp/"$1"_evt/"$2"/"$split".mrp


    python mtool/main.py --strings --ids --read ace --write mrp "$indata" "$outdata"
done;


