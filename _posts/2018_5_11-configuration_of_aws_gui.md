---
layout: post
title: configuration_of_aws_gui
---
# enable password login
sudo sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo /etc/init.d/ssh restart
sudo passwd ubuntu

# enable GUI
sudo apt-get install xrdp
sudo apt-get install xfce4
sudo echo xfce4-session>~/.xsession  
sudo service xrdp restart

# reference
* https://aws.amazon.com/ko/premiumsupport/knowledge-center/connect-to-ubuntu-1604-windows/
* http://hellogohn.com/post_one192
