#!/bin/bash
# 获取2018-19赛季数据
mkdir -p data/raw/2018-19
wget https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2018-19/players_raw.csv -O data/raw/2018-19/players.csv
wget https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2018-19/understat/understat_player.csv -O data/2018-19/understat.csv 