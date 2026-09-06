#!/bin/bash
cd "$(dirname "$0")"
# wait for the stage to finish (either the OK line lands, or the stager exits)
until grep -qE "OK: all|ranks staged" stage_flash.log 2>/dev/null || ! pgrep -f run_ds4f_stage_11n.sh >/dev/null 2>&1; do
    sleep 30
done
sleep 5
if ! grep -qE "OK: all|ranks staged" stage_flash.log 2>/dev/null; then
    echo "STAGE FAILED -- not running A/B" > ab_scanmin_flash.txt
    tail -5 stage_flash.log >> ab_scanmin_flash.txt
    exit 1
fi
echo "stage OK -> running A/B" >> stage_flash.log
./ab_scanmin_flash.sh > ab_scanmin_flash_outer.log 2>&1
