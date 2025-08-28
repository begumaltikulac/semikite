#!/bin/sh -x
GREEN=9
RED=11

# gpio setup
for pin in $GREEN $RED;do
    echo $pin > /sys/class/gpio/export
    echo out  > /sys/class/gpio/gpio$pin/direction
    echo 0    > /sys/class/gpio/gpio$pin/value
done

greenon  () { 
    echo 1 > /sys/class/gpio/gpio$GREEN/value
    }
greenoff () { 
    echo 2 > /sys/class/gpio/gpio$GREEN/value 
    }
redon    () { 
    echo 1 > /sys/class/gpio/gpio$RED/value
    }
redoff   () { 
    echo 0 > /sys/class/gpio/gpio$RED/value 
    }

IP=192.168.206.50
LOGFILE=/home/pi/cam/camfetch.log

if ping $IP -c1 -W5;then
    greenon
else
    redon
    echo "$(date %F-%T): could not ping $IP" | tee -a $LOGFILE
    exit 1
fi

if wget "$IP/cgi-bin/viewer/video.jpg" --timeout=20 --user=root --password='ndr!min' -O "/home/pi/cam/img/$(date +Image_%Y%m%d_%H%M%S.jpg)";then
    greenon
else
    redon
    echo "$(date %F-%T): could not wget $IP/cgi-bin/viewer/video.jpg" | tee -a $LOGFILE
    exit 1
fi
