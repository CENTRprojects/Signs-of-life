"""Saving outputs to database for both the crawler and Labtool samples"""
import base64
from config import RUN_CONFIG
import csv
from formatting import TABLEAU_COLUMNS 
import psycopg2 as pg
import sys
from os import listdir
from os.path import isfile, join, dirname, abspath, basename
from pathlib import Path
import requests
import shutil
from datetime import date
import json
from multiprocessing.dummy import Pool as ThreadPool
# performance logger
from utils import PerformanceLogger
plog = PerformanceLogger(filepath=join(RUN_CONFIG["MAIN_DIR"],"logging"), filename="db_perf.log", enable_logging = RUN_CONFIG['PERFORMANCE_LOGGING'])
sam_plog = PerformanceLogger(filepath=join(RUN_CONFIG["MAIN_DIR"],"logging"), filename="sam_p.log", enable_logging = RUN_CONFIG['PERFORMANCE_LOGGING'])
sep = RUN_CONFIG["CSV_OUTPUT_DELIMITER"]


DB_FIELDS = set()
db = None
input_urls = set()

def executeSQLFromFile(filename):
	with open(filename, 'r') as f:
		sqlCommands = f.read().replace('\n', '').split(";")
	cur = db.cursor()
	for command in sqlCommands:
		if len(command) > 0:
			try:
				cur.execute(command)
			except pg.ProgrammingError:
				db.rollback()
				print(f"Error running command : {command}")
				raise
	cur.close()
	db.commit()
	
def ConnectDB():
	print("\nConnecting to DB...")
	global db
	try: 
		# first check the db exists
		db = pg.connect(host=RUN_CONFIG["DBHOST"], port=RUN_CONFIG["DBPORT"], database=RUN_CONFIG["DBNAME"], user=RUN_CONFIG["DBUSER"], password=RUN_CONFIG["DBPASS"])
		cur = db.cursor()
		# then check the table exists
		try:
			cur.execute("select * from signs_of_life_crawler limit 1")
		except (Exception, pg.ProgrammingError) as error:
			# if we're here it means we connected to the db but there is no table
			plog.it(f"Error connecting to table: {error}", is_error=True)
			db.rollback()
			sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_1.sql")
			executeSQLFromFile(sqlfile)
			sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_2.sql")
			executeSQLFromFile(sqlfile)
			sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_3.sql")
			executeSQLFromFile(sqlfile)
			sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_4.sql")
			executeSQLFromFile(sqlfile)
			cur.execute("select * from signs_of_life_crawler limit 1")
		try:
			# check the sample update has been applied
			cur.execute("select clean_filename from signs_of_life_crawler limit 1")
		except (Exception, pg.ProgrammingError) as error:
			plog.it(f"Error connecting to table: {error}", is_error=True)
			db.rollback()
			sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_5.sql")
			executeSQLFromFile(sqlfile)
			cur.execute("select to_sample from signs_of_life_crawler limit 1")
		try:
			# check the fx_hcj__* fields have been added to the db
			cur.execute("select fx_hcj__secu_xss from signs_of_life_crawler limit 1")
		except (Exception, pg.ProgrammingError) as error:
			plog.it(f"Error connecting to table: {error}", is_error=True)
			db.rollback()
			sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_6.sql")
			executeSQLFromFile(sqlfile)
			cur.execute("select to_sample from signs_of_life_crawler limit 1")
		q = cur.fetchall()
		cur.close()
		print("DB Connection successful.")
	except (Exception, pg.Error) as error:
		plog.it(f"Error while connecting to PostgreSQL: {error}", is_error=True)
		raise

def DiscoverFiles():
	print("\nLooking for output files...")
	csvpath = dirname(RUN_CONFIG["output_final_file_path"])
	plog.it("CSVPath: " + csvpath)
	outputfiles=[join(csvpath,f) for f in listdir(csvpath) if isfile(join(csvpath, f))]
	print("Found {} output files.".format(len(outputfiles)))
	return outputfiles

def ProcessFiles(outputfiles):
	print("\nScanning files...")
	for file in outputfiles:
		print("Checking {}".format(file))
		if "final_" in file and "_DEBUG" not in file and "_tempo" not in file:
			print("\n\tPROCESSING FILE: {}".format(file))
			ProcessFile(file)
	print("Finished scanning all files.")

def ConvertBool(csvbit):
	"""csv is sending in various values for bool (eg 1.0, "1.0", True and so on), but python is converting everything to true and null to false.
	Have to parse it properly before handing it over to psycopg2 to insert into postgres"""
	if csvbit == "":
		return "null"
	else:
		return csvbit
	#elif csvbit in [1.0,1,True,"1.0","1","true","True"]:
	#	return True
	#elif csvbit in [0.0,0,False,"0.0","0","false","False"]:
	#	return False
	#elif type(csvbit) == type(''):
	#	return csvbit
	raise ValueError('Unknown boolean type: {}({})'.format(csvbit, type(csvbit)))
	
def FilterColumns():
	""" we don't want everything from the debug columns to be stored in the database, so we'll define what we want to keep (or reject) here.
	"""
	global DB_FIELDS
	for col in TABLEAU_COLUMNS:
		# add rejected columns to the list here
		if col not in ["target_url", "url", "registrar_found", "kw_parked", "pred_is_empty", "js_or_iframe_found", "original_url", "category", "subcategory", "pred_is_parked", "is_error", "kw_park_notice", "original_url"]:
			DB_FIELDS.add(col)
	plog.it("TABLEAU_COLUMNS: {}".format(TABLEAU_COLUMNS))
	plog.it("DB_FIELDS: {}".format(DB_FIELDS))
	missing = set(TABLEAU_COLUMNS) - set(DB_FIELDS)
	if len(missing) > 0:
		plog.it(f"These are missing from DB_FIELDS:{set(TABLEAU_COLUMNS)- set(DB_FIELDS)} ")

def ConstructSQLBase():
	"""
		Get the columns from formatting.py and use that to construct the insert	
	"""
	global DB_FIELDS
	msg = "insert into signs_of_life_crawler (filename, "
	for col in DB_FIELDS:
		msg += col.lower() + ", "
	msg = msg[:-2] + ") values "
	return msg

def ConstructSQLData(filename, row):
	"""
		Extract the data from the csv row (dict) and construct a string to be added to the qstr
	"""
	global DB_FIELDS, input_urls
	#if 'to_sample' in row and row['to_sample'] is None:
	#	sam_plog.it(f"Found error row: {row}", is_error=True)
	#	return ''
	msg = "($_${}$_$,".format(filename)
	for col in DB_FIELDS:
		if col == 'input_url':
			if row[col] in input_urls:
				plog.it(f'Found this input_url twice (there should only be one): {ConvertBool(row[col])}')
			else:
				input_urls.add(row[col])
		if col == 'to_sample' and row['to_sample'] in [None, "null"]:
			row["to_sample"] = "False"

		if col in ['ss_filename', 'raw_filename', 'clean_filename', 'json_filename']:
			if 'to_sample' not in row or row['to_sample'] in [None, "null"]:
				row['to_sample'] = "False"
			if row['to_sample'] == "False":
				msg += "$_$null$_$,"
			else:
				msg += "$_${}$_$,".format(row[col])
		else:
			try:
				msg += "$_${}$_$,".format(ConvertBool(row[col]))
			except KeyError:
				msg += "$_${}$_$,".format('null')
			
	msg = msg[:-1] # get rid of extra comma
	msg += "), "
	return msg

def ConstructSQLTail(qstr):
	"""
		Get the columns from DB_FIELDS and construct the on conflict rules
	"""
	qstr = qstr[:-2]
	# msg_tail = " on conflict (filename,date,input_url) do update set filename = excluded.filename, "
	#for col in DB_FIELDS:
	#	msg_tail += f"{col} = excluded.{col}, "
	#msg_tail = msg_tail[:-2]
	msg_tail = " on conflict (filename,date,input_url) do nothing"
	return qstr+msg_tail

def ProcessFile(file):
	# get a good filename, check if we're trying to input this data again (due to some error)
	base_filename = basename(file)
	if "-DB_" not in file:
		filename = date.today().strftime("%Y-%m-%d")+"-DB_"+base_filename
	else:
		filename = base_filename
	plog.it("filename: " + filename)
	destination = join(dirname(dirname(abspath(__file__))), "completed", filename)
	plog.it("destination: "+ destination)
	
	# Setup the query string that we'll use to input into the database
	qlist = []
	qbase = ConstructSQLBase()
	qstr = qbase
	# Open file, read it, process it.
	plog.it("input file: "+ file)
	plog.it("qstr: {}".format(qstr))
	samples = []
	with open(file, newline='', encoding="utf-8-sig") as csvfile:
		reader = csv.DictReader(csvfile, delimiter=sep)
		for row in reader:
			#plog.it(row)
			if len(qstr) > 10000:
				qlist.append(ConstructSQLTail(qstr))
				qstr = qbase
			# add to the query_string for this row
			# check domain isn't nan
			qstr += ConstructSQLData(filename, row)
			
			# check if row is a sample for LabTools, add a representational dict to samples[] with the relevant filenames
			if 'to_sample' in row and row["to_sample"] == "True" and RUN_CONFIG['DO_SAMPLING']:
				sample = row.copy()
				sample['url'] = sample['input_url'] if sample['input_url'].startswith('http') else 'http://' + sample['input_url']
				samples.append(sample)
		qlist.append(ConstructSQLTail(qstr))
	# check we're not duplicating the tlds we found with the filename
	#CheckDB(filename)
	# when all looks good, submit it to the database
	plog.it("qstr: "+ qstr)
	plog.it("qlist len: {}".format(len(qlist)))
	WriteToDB(qlist)
	# then move the file to the completed folder
	try:
		shutil.move(file, destination)
	except (Exception) as error:
		print(error)
		input("Close the csv if it is open and press a key to try again.")
		try:
			shutil.move(file, destination)
		except (Exception) as error1:
			plog.it(str(error1.msg), is_error=True)
			plog.it("Critical failure...I'm giving up", is_error=True)
	# process samples
	if len(samples) > 0:
		ProcessSamples(samples)

def get_base64_encoded_image(image_path):
	with open(image_path, 'rb') as img_file:
		return base64.b64encode(img_file.read()).decode('utf-8')
def get_text_file_contents(text_file_path):
	with open(text_file_path, 'r') as txt_file:
		return txt_file.read().encode('utf-8')

def create_categorisation_json(sample, file_data):
	#sam_plog.it(f"Create JSON Categorisation data | sample : {type(sample)} | file_data : {type(file_data)}")
	json_data = json.loads(file_data)
	#sam_plog.it(f"\tAfter loading | sample : {type(sample)} | json_data : {type(json_data)}")
	for k,v in json_data.items():
		if k not in sample:
			sample[k] = v	
	return json.dumps(sample)

def UploadSample(sample):
	if sample['ss_filename'] in [None, '']:
		return
	domain = sample['url']
	try:
		# grab session to use
		session = get_session()
		image_upload_url = RUN_CONFIG["SAMPLING_REST_API_IMAGE_URL"]
		sample_upload_url = RUN_CONFIG["SAMPLING_REST_API_SAMPLE_URL"]
		updating_allowed = RUN_CONFIG["SAMPLING_REST_API_ALLOW_UPDATES"]
		auth_check = session.get(image_upload_url)
		sam_plog.it(f"({domain}) | Auth result : {auth_check.status_code} | {auth_check.content[:100]}")
		sam_plog.it(f"({domain}) | CSRF Token : {auth_check.cookies['csrftoken']}")

		# get filename ready
		sample_filename = Path(sample['ss_filename']).absolute()
		file_path = Path(RUN_CONFIG['SAMPLING_LOCAL_FOLDER']).joinpath(sample_filename) # construct absolute url for this image

		sam_plog.it(f"({domain}) | sample_filename : {sample_filename}\n\t\t | file path : {file_path}\n\t\t | post_url : {image_upload_url}")
		
		# get data ready: filename (str), screenshot(base64). Title will be filename minus extension.
		data = {'filename':file_path.name, 'screenshot':get_base64_encoded_image(file_path)}

		r = session.post(image_upload_url, data=data)
		sam_plog.it(f"({domain}) | Image POST response : {r.status_code} | {r.content[:100]}")
		if r.status_code == 200:
			sample['screenshot'] = int(r.content) # pk for uploaded image
		else:
			msg = f"ERROR: could not upload image to {image_upload_url}.\n\nReceived status code: {r.status_code}\nReceived msg: {r.content}\n\n"
			return msg
						
	except Exception as e:
		msg = f"({domain}) | ERROR: could not process this screenshot properly : {sample_filename} | reason : {e}"
		return msg	
	try:
		# upload the data
		data = {
			'url': sample['url'],
			'html_text': get_text_file_contents(sample['clean_filename']),
			'html_raw': get_text_file_contents(sample['raw_filename']),
			'categorisation': create_categorisation_json(sample, get_text_file_contents(sample['json_filename'])),
			'screenshot': sample['screenshot'],
		}
		r = session.post(sample_upload_url, data=data)
		sam_plog.it(f"({domain}) | Sample POST response : {r.status_code} | url : {data['url']}")
		if r.status_code != 201:
			if r.status_code == 200: # we've been redirected to the Update url for this sample, update sample based on global flag
				sam_plog.it(f"\t({domain}) | Redirected to : {r.url} | Patching? {updating_allowed}")
				if updating_allowed:
					r = session.patch(r.url, data=data)
					sam_plog.it(f"\t({domain}) | Patch Response: {r.status_code} | {r.content[:1000]}")
			else:
				msg = f"\n({domain}) | ERROR: could not upload sample to {sample_upload_url}.\n\nReceived status code: {r.status_code}\nReceived msg: {r.content[:100]}\n\n"
				return msg
	except FileNotFoundError as e:
		msg = f"({domain}) | ERROR: could not find a file ({e}) for this sample: {sample}"
		return msg

def get_session():
	user_auth = (RUN_CONFIG["SAMPLING_REST_API_USERNAME"], RUN_CONFIG["SAMPLING_REST_API_PASSWORD"])
	#sam_plog.it(f"user_auth : {user_auth}")
	session = requests.Session()
	session.auth = user_auth
	return session

def ProcessSamples(samples):
	sam_plog.it(f"Processing {len(samples)} samples")
	if RUN_CONFIG['MULTI_PROCESSING']:
		pool = ThreadPool(processes=RUN_CONFIG['WORKERS_POST_PROCESSING'])
		results = pool.map(UploadSample, samples)
		pool.close()
		pool.join()
		pool.terminate()
		results = [r for r in results if r != None]
	else:
		results = []
		for sample in samples:
			r = UploadSample(sample)
			if r != None:
				results.append(r)
	sam_plog.it(f"SAMPLE PROCESSING COMPLETE")
	if len(results) > 0:
		sam_plog.it(f"FOUND {len(results)} ERRORS: ")
		for r in results:
			sam_plog.it(r)
		
def WriteToDB(qlist):
	print("\tWriting to database")
	plog.it("There are {} qstr in qlist".format(len(qlist)))
	for qstr in qlist:
		try:
			cur = db.cursor()
			cur.execute(qstr)
			db.commit()
			cur.close()
		except (Exception, pg.DatabaseError) as error:
			plog.it(error, is_error=True)
			plog.it(f"Error string: {qstr}", is_error=True)
			db.rollback()
			raise
	print("\tDatabase written")

def DoIt(filename=None):
	plog.perf_go(f"Starting output processing | filename : {filename}")
	if filename is None:
		outputFiles = DiscoverFiles()
	else:
		outputFiles = [filename]
	if len(outputFiles) > 0:
		plog.it("Connecting to DB.")
		ConnectDB()
		plog.it("Filtering Columns.")
		FilterColumns()
		plog.it("Processing Files")
		ProcessFiles(outputFiles)
		plog.it("Closing DB connection")
		if(db):
			db.close()
	else:
		plog.it("\nNo output files found. Have you run the rest of the program?")
	plog.perf_end("Completed output processing")

if __name__ == '__main__':
	if len(sys.argv) > 1:
		DoIt(sys.argv[1])
	else:
		DoIt()
