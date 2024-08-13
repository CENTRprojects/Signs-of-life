#!/bin/bash
#
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
#
PROCCESS_DIR='data/*.csv'

CONTAINER_NAME=$(hostname)

crawl_dir () {
    for file in $PROCCESS_DIR
    do
        echo -e "\n\n*****************\n\n"
        echo -e "* ${BLUE}Input file found [${file}]${NC}";
        #
        if [ "$(sed -n '/^url/p;q' "${file}")" ] 
        then
            echo -e "* ${GREEN}found URL in file [${file}] - Crawling.${NC}"
            BASENAME=$(basename "$file")
            mkdir -p $input_folder
            mkdir -p $input_folder/${CONTAINER_NAME}
            mv "$file" $input_folder/${CONTAINER_NAME}
            python app_domains/main_domains.py -container_name="${CONTAINER_NAME}"
            mkdir -p done
            mv "$input_folder/${CONTAINER_NAME}/$BASENAME" done/
            rm  $input_folder/${CONTAINER_NAME}/$BASENAME.chunk*
        else
            echo -e "* ${RED}no URL found in file [${file}] - skipping.${NC}"
        fi
    done
    echo -e "\n\n*****************\n\n"
}

main () {
   echo -e "* ${BLUE}Looking for input files to Crawl in [${PROCCESS_DIR}].${NC}";
   crawl_dir
   echo -e "* ${BLUE}Done Crawling. Going to Sleep.....${NC}";
   sleep 60s  
   main
}

echo -e "* ${BLUE}Starting the Crawler. ID: ${CONTAINER_NAME}${NC}";
main
echo -e "* ${BLUE}Bye Bye....!${NC}";

#ping 127.0.0.1