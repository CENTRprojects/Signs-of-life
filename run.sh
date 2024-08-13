#!/bin/bash
./report_input.sh
echo Starting network logging
echo nload
echo Removing screen log
rm screenlog.0
echo Running Crawler
python app_domains/main_domains.py
