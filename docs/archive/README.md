Signs of Life Crawler

Take a big list of domains, visit them, try and figure out it's being used for, if anything.

Installation:

Signs of Life crawler requires the Anaconda Platform for python3+: https://www.anaconda.com/distribution/
- Install anaconda3 in a user writeable space, make sure you have the correct python version that matches it. 
- Once installed, you need to run the init on every account you want to use it with -> /opt/anaconda3/bin/conda init
- by default it will enable the base shell for every login. To disable this behaviour, type conda config --set auto_activate_base False. You will still be able to use conda commands in shell if you do this. To activate the shell manually, type: source <install dir>/bin/activate
- Install virtual box via download: https://www.virtualbox.org/wiki/Linux_Downloads


-- Then run the following from the conda shell:

conda install aiohttp asn1crypto async-timeout attrs beautifulsoup4 certifi cffi chardet conda cryptography idna joblib menuinst multidict nltk numpy pandas pycosat pycparser pyOpenSSL PySocks python-dateutil pytz pywin32 requests scipy singledispatch six soupsieve tqdm urllib3 wincertstore yarl
conda install -c conda-forge ruamel.yaml cchardet
pip install -i https://pypi.anaconda.org/rsmulktis/simple bs4 
pip install idna-ssl win-inet-pton pycares asyncpool typing-extensions recommendation-system jieba pyvbox asyncio langdetect aiodns psycopg2


Running App:

1) Edit app_domains/config.py to modify program settings such as maximum connections, workers, threads and so on.
6) .env file should be in the root directoy, and settings should be of the format: setting=value
2) Put a list of domains you want to check into a csv file, and save it to input/batch_test. There can be multiple files.
3) Run python app_domains/main_domains.py.
4) Results are stored in the output folder and completed folder.
5) If a postgresql database is defined in .env or config.py, it will process all DEBUG csvs in the output folder, then move those files to the completed folder.
 - The data will be stored in a signs_of_life_crawler table in the DBNAME database.
