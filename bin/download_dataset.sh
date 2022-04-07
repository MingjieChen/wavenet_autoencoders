#!/bin/bash

PASSWORD=XXXX_REPLACE_WITH_THE_PASSWORD_XXXX
for ext in zip z01 z02
do
    wget https://download.zerospeech.com/2020/zerospeech2020.$ext || exit 1
done
exit 0
